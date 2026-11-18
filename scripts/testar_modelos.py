"""
testar_modelos.py - Testador visual comparativo (YOLOv8 vs Faster R-CNN).
Arquiva resultados anteriores em archive/records/ e salva os novos em resultados/.
"""

import os
import shutil
import datetime
from pathlib import Path
from typing import List, Dict

import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DIR_IMAGENS = ROOT / "imagens_teste"
DIR_RESULTADOS = ROOT / "resultados"
DIR_RECORDS = ROOT / "archive" / "records"
DIR_WEIGHTS = ROOT / "modelos" / "weights"

CLASS_NAMES = {1: "D00 - Long", 2: "D10 - Trans", 3: "D20 - Jacare", 4: "D40 - Buraco"}
CONF_MIN = 0.05


def to_long_path(p: Path) -> str:
    """Suporte a caminhos longos no Windows."""
    s = str(p.resolve())
    return f"\\\\?\\{s}" if (os.name == "nt" and not s.startswith("\\\\?\\")) else s


def arquivar_resultados():
    """Move resultados de resultados/ para archive/records/ sem sobrescrever."""
    if not DIR_RESULTADOS.exists():
        DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
        return
    arquivos = [f for f in DIR_RESULTADOS.iterdir() if f.is_file()]
    if not arquivos:
        return

    DIR_RECORDS.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for f in arquivos:
        dst = DIR_RECORDS / f.name
        if dst.exists():
            dst = DIR_RECORDS / f"{f.stem[:80]}_{ts}{f.suffix}"
        shutil.move(to_long_path(f), to_long_path(dst))
    print(f"Resultados anteriores arquivados em: {DIR_RECORDS.relative_to(ROOT)}")


def carregar_modelos() -> Dict[str, object]:
    """Carrega os pesos de YOLOv8 e Faster R-CNN."""
    # YOLO
    yolo_dir = DIR_WEIGHTS / "yolo"
    yolo_ckpt = yolo_dir / "best.pt" if (yolo_dir / "best.pt").is_file() else (yolo_dir / "last.pt" if (yolo_dir / "last.pt").is_file() else yolo_dir / "yolov8s.pt")
    if not yolo_ckpt.is_file():
        raise FileNotFoundError("Nenhum peso YOLO encontrado em modelos/weights/yolo/")
    yolo = YOLO(str(yolo_ckpt))

    # Faster R-CNN
    faster_dir = DIR_WEIGHTS / "faster"
    if not faster_dir.is_dir() and (DIR_WEIGHTS / "faster r-cnn").is_dir():
        faster_dir = DIR_WEIGHTS / "faster r-cnn"
    faster_ckpt = faster_dir / "modelo_ultimo.pth"

    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    faster = fasterrcnn_resnet50_fpn(weights=None)
    faster.roi_heads.box_predictor = FastRCNNPredictor(faster.roi_heads.box_predictor.cls_score.in_features, 5)
    if faster_ckpt.is_file():
        faster.load_state_dict(torch.load(str(faster_ckpt), map_location="cpu", weights_only=False))
    faster.eval()

    return {"yolo": yolo, "faster": faster}


def detectar(img: Image.Image, modelo, is_yolo: bool) -> List[Dict]:
    """Inferência em uma imagem com o modelo especificado."""
    dets = []
    if is_yolo:
        res = modelo(img, conf=CONF_MIN, verbose=False)
        for r in res:
            for i in range(len(r.boxes)):
                c = int(r.boxes.cls[i]) + 1
                dets.append({"box": r.boxes.xyxy[i].tolist(), "label": CLASS_NAMES.get(c, f"C{c}"), "score": float(r.boxes.conf[i])})
    else:
        with torch.no_grad():
            outs = modelo(F.to_tensor(img).unsqueeze(0))[0]
        for b, s, c in zip(outs["boxes"].tolist(), outs["scores"].tolist(), outs["labels"].tolist()):
            if s >= CONF_MIN:
                dets.append({"box": b, "label": CLASS_NAMES.get(int(c), f"C{int(c)}"), "score": s})
    return dets


def desenhar_comparativo(img_path: Path, det_yolo: List[Dict], det_faster: List[Dict]):
    """Desenha detecções lado a lado: YOLO (vermelho) à esquerda, Faster (azul) à direita."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w * 2, h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    canvas.paste(img, (w, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    def desenhar(lista, offset, cor):
        for d in lista:
            x1, y1, x2, y2 = d["box"]
            box = [x1 + offset, y1, x2 + offset, y2]
            draw.rectangle(box, outline=cor, width=2)
            txt = f"{d['label']} {d['score']:.2f}"
            bb = draw.textbbox((0, 0), txt, font=font)
            draw.rectangle([box[0], box[1] - (bb[3]-bb[1]) - 4, box[0] + (bb[2]-bb[0]) + 4, box[1]], fill=cor)
            draw.text((box[0] + 2, box[1] - (bb[3]-bb[1]) - 2), txt, fill="white", font=font)

    desenhar(det_yolo, 0, "red")
    desenhar(det_faster, w, "blue")
    draw.text((10, 10), "YOLOv8", fill="red", font=font)
    draw.text((w + 10, 10), "Faster R-CNN", fill="blue", font=font)

    canvas.save(to_long_path(DIR_RESULTADOS / f"comparativo_{img_path.stem}.jpg"), quality=90)


def main():
    arquivar_resultados()
    modelos = carregar_modelos()

    exts = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
    imagens = sorted([p for p in DIR_IMAGENS.iterdir() if p.suffix.lower() in exts])
    if not imagens:
        print("Nenhuma imagem encontrada em imagens_teste/")
        return

    print(f"Processando {len(imagens)} imagens de teste...")
    for img_path in imagens:
        img_rgb = Image.open(img_path).convert("RGB")
        desenhar_comparativo(img_path, detectar(img_rgb, modelos["yolo"], True), detectar(img_rgb, modelos["faster"], False))

    manifesto = DIR_RESULTADOS / "manifest.txt"
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(to_long_path(manifesto), "w", encoding="utf-8") as f:
        f.write(f"Teste Comparativo: {agora}\nTotal Imagens: {len(imagens)}\n")

    print(f"Concluído! {len(imagens)} comparativos salvos em: {DIR_RESULTADOS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
