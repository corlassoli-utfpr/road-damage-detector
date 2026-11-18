"""
treinar.py - Treinador para YOLOv8 e Faster R-CNN (RDD2022).
- Checkpoints salvos a CADA ÉPOCA em modelos/weights/ para segurança contra interrupções.
- Retomada inteligente: calcula automaticamente as épocas restantes sem necessidade de contas.
- Controle térmico ativo por batch a cada 50 passos.
"""

import os
import sys

# Desativa GPU se CPU for solicitada antes de carregar torch/ultralytics
if "--device" in sys.argv:
    idx = sys.argv.index("--device")
    if idx + 1 < len(sys.argv) and sys.argv[idx + 1].lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import datetime
import shutil
import subprocess
import time
import tempfile
from pathlib import Path
import random
import yaml
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent
DIR_DATASET = ROOT / "dataset_rdd"
DIR_WEIGHTS = ROOT / "modelos" / "weights"
DIR_MANIFESTS = ROOT / "modelos" / "manifests"
DIR_TEMP = Path(tempfile.gettempdir()) / "rdd_treino_temp"

MAPA = {0: 0, 1: 1, 2: 2, 4: 3}
NOMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Configurações CLI
parser = argparse.ArgumentParser(description="Treino YOLOv8 / Faster R-CNN com checkpoints seguros")
parser.add_argument("target", nargs="?", default="yolo", choices=["yolo", "faster"], help="Modelo: yolo ou faster")
parser.add_argument("epochs", type=int, nargs="?", default=10, help="Épocas (padrão: 10)")
parser.add_argument("batch", type=int, nargs="?", default=8, help="Batch size (padrão: 8)")
parser.add_argument("limit", type=int, nargs="?", default=3000, help="Limite de imagens (Faster)")
parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="cpu ou cuda")
parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
parser.add_argument("--novo", action="store_true", help="Inicia modelo do zero (ignora checkpoints)")
parser.add_argument("--gpu-frac", type=float, default=0.85, help="Fração de memória GPU")
parser.add_argument("--threads", type=int, default=4, help="Threads da CPU")
parser.add_argument("--temp-limit", type=float, default=76.0, help="Limite térmico da GPU em °C (padrão: 76)")
parser.add_argument("--workers", type=int, default=2, help="Workers do DataLoader")
args = parser.parse_args()

torch.set_num_threads(args.threads)
if args.device == "cuda" and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(args.gpu_frac, 0)
disp = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

contador_batch = 0


def checar_temp(forcar: bool = False):
    """Monitora a GPU e pausa até esfriar se ultrapassar o limite térmico."""
    if args.device != "cuda" or not args.temp_limit:
        return
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True)
        temp = float(out.strip().split("\n")[0])
        if temp >= args.temp_limit:
            alvo = args.temp_limit - 6.0
            print(f"\n[ALERTA TÉRMICO] GPU a {temp:.0f}°C (limite: {args.temp_limit:.0f}°C). Pausando até {alvo:.0f}°C...")
            while temp > alvo:
                time.sleep(15)
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True)
                temp = float(out.strip().split("\n")[0])
            print(f"[RESFRIADA] GPU a {temp:.0f}°C. Retomando treino...\n")
    except Exception:
        pass


def monitor_batch(trainer):
    """Checa temperatura a cada 50 batches."""
    global contador_batch
    contador_batch += 1
    if contador_batch % 50 == 0:
        checar_temp()


def registrar_manifesto(tipo: str, detalhes: str = ""):
    """Atualiza o manifesto correspondente."""
    pasta = DIR_MANIFESTS / tipo
    pasta.mkdir(parents=True, exist_ok=True)
    manifest = pasta / "manifest.txt"
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(f"\nTreino: {agora} | Épocas: {args.epochs} | Batch: {args.batch} | Dispositivo: {args.device} {detalhes}\n")


def limpar():
    """Remove pastas temporárias."""
    for p in (DIR_TEMP, ROOT / "runs", ROOT / "treinos", ROOT / ".treino_temp"):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def preparar_dados():
    """Garante existência do data.yaml."""
    yaml_file = DIR_DATASET / "data.yaml"
    if not yaml_file.exists():
        DIR_DATASET.mkdir(parents=True, exist_ok=True)
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({"path": str(DIR_DATASET.resolve()), "train": "train/images", "val": "val/images", "names": NOMES}, f)
    return yaml_file


def treinar_yolo(yaml_file):
    """Treinamento do YOLOv8 com retomada inteligente e salvamento a cada época."""
    from ultralytics import YOLO

    pasta_yolo = DIR_WEIGHTS / "yolo"
    pasta_yolo.mkdir(parents=True, exist_ok=True)
    ckpt_last = pasta_yolo / "last.pt"
    modelo_base = pasta_yolo / "yolov8s.pt"

    resume = False
    if not args.novo and ckpt_last.is_file():
        model = YOLO(str(ckpt_last))
        # Inspeciona o checkpoint para saber a época onde parou
        try:
            ckpt_info = torch.load(str(ckpt_last), map_location="cpu", weights_only=False)
            ep_concluida = ckpt_info.get("epoch", -1) + 1  # 0-indexed na Ultralytics
            if ep_concluida >= args.epochs:
                # Se as épocas pedidas já foram concluídas, adiciona mais épocas sobre o total
                meta_epochs = ep_concluida + args.epochs
                print(f"✔ Checkpoint com {ep_concluida} épocas concluídas. Adicionando mais {args.epochs} épocas (Meta: {meta_epochs} épocas)...")
                args.epochs = meta_epochs
            elif ep_concluida > 0:
                print(f"✔ Retomando YOLOv8 da época {ep_concluida + 1} até a meta de {args.epochs} épocas...")
            else:
                print(f"✔ Retomando YOLOv8 de: {ckpt_last.name} (Meta: {args.epochs} épocas)")
            resume = True
        except Exception:
            resume = True
            print(f"✔ Retomando YOLOv8 a partir de: {ckpt_last.name}")
    else:
        origem = str(modelo_base) if modelo_base.is_file() else "yolov8s.pt"
        print(f"✔ Iniciando NOVO YOLOv8 ({args.epochs} épocas) a partir de: {origem}")
        model = YOLO(origem)
        resume = False

    DIR_TEMP.mkdir(parents=True, exist_ok=True)

    def salvar_checkpoint_epoca(trainer):
        checar_temp(forcar=True)
        pesos_dir = Path(trainer.save_dir) / "weights"
        for n in ("best.pt", "last.pt"):
            if (pesos_dir / n).is_file():
                shutil.copy2(pesos_dir / n, pasta_yolo / n)
        ep = getattr(trainer, "epoch", 0) + 1
        print(f"[CHECKPOINT] Época {ep}/{args.epochs} salva em: {pasta_yolo.relative_to(ROOT)}/last.pt")

    model.add_callback("on_train_batch_end", monitor_batch)
    model.add_callback("on_train_epoch_end", salvar_checkpoint_epoca)

    try:
        model.train(
            data=str(yaml_file),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=str(DIR_TEMP),
            name="yolo",
            exist_ok=True,
            resume=resume,
        )
        registrar_manifesto("yolo", f"| ImgSize: {args.imgsz}")
        print(f"Treino concluído com sucesso! Pesos em: {pasta_yolo.relative_to(ROOT)}")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPÇÃO] Treino pausado. Último checkpoint preservado em: {pasta_yolo.relative_to(ROOT)}/last.pt")
        print("Para retomar depois, basta executar sem a flag '--novo'.")
    finally:
        limpar()


def treinar_faster():
    """Treinamento do Faster R-CNN com salvamento a cada época."""
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.functional as TF
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    class RDDDataset(Dataset):
        def __init__(self, limite):
            imgs = [p for p in (DIR_DATASET / "train" / "images").iterdir() if p.suffix.lower() in EXTS]
            validas = [p for p in imgs if (DIR_DATASET / "train" / "labels" / f"{p.stem}.txt").exists()]
            random.seed(42)
            self.imgs = random.sample(validas, min(limite, len(validas)))

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            p = self.imgs[idx]
            im = Image.open(p).convert("RGB")
            w, h = im.size
            boxes, labels = [], []
            for ln in (DIR_DATASET / "train" / "labels" / f"{p.stem}.txt").read_text().splitlines():
                partes = ln.split()
                if partes:
                    c, cx, cy, bw, bh = map(float, partes[:5])
                    x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                    x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                    if x2 > x1 + 1 and y2 > y1 + 1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(c) + 1)
            return TF.to_tensor(im), {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}

    pasta_faster = DIR_WEIGHTS / "faster"
    pasta_faster.mkdir(parents=True, exist_ok=True)
    ckpt = pasta_faster / "modelo_ultimo.pth"

    if not args.novo and ckpt.is_file():
        print(f"✔ Retomando Faster R-CNN a partir de: {ckpt.name} ({args.epochs} épocas)")
        model = fasterrcnn_resnet50_fpn(weights=None, min_size=args.imgsz, max_size=args.imgsz)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 5)
        model.load_state_dict(torch.load(ckpt, map_location=disp, weights_only=False))
    else:
        print(f"✔ Iniciando NOVO Faster R-CNN ({args.epochs} épocas)")
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=args.imgsz, max_size=args.imgsz)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 5)

    model.to(disp)
    otimizador = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(disp.type == "cuda"))
    loader = DataLoader(RDDDataset(args.limit), batch_size=args.batch, shuffle=True, collate_fn=lambda b: tuple(zip(*b)), num_workers=args.workers)

    try:
        for ep in range(1, args.epochs + 1):
            model.train()
            loss_total = 0.0
            step = 0
            for imgs, tgts in loader:
                step += 1
                if step % 50 == 0:
                    checar_temp()
                imgs = [i.to(disp) for i in imgs]
                tgts = [{k: v.to(disp) for k, v in t.items()} for t in tgts]
                otimizador.zero_grad()
                with torch.amp.autocast("cuda", enabled=(disp.type == "cuda")):
                    loss = sum(model(imgs, tgts).values())
                scaler.scale(loss).backward()
                scaler.step(otimizador)
                scaler.update()
                loss_total += loss.item()

            # Salva o checkpoint ao final de CADA época
            torch.save(model.state_dict(), ckpt)
            print(f"Época {ep}/{args.epochs} - Loss: {loss_total / len(loader):.4f} [Checkpoint salvo]")

        registrar_manifesto("faster", f"| Limite: {args.limit}")
        print(f"Treino concluído. Salvo em: {pasta_faster.relative_to(ROOT)}/{ckpt.name}")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPÇÃO] Treino pausado. Último checkpoint salvo em: {pasta_faster.relative_to(ROOT)}/{ckpt.name}")
        print("Para retomar depois, basta executar sem a flag '--novo'.")
    finally:
        limpar()


if __name__ == "__main__":
    limpar()
    yaml_file = preparar_dados()
    if args.target == "yolo":
        treinar_yolo(yaml_file)
    else:
        treinar_faster()
