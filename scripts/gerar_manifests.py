"""
gerar_manifests.py - Gera manifestos dos modelos em modelos/weights/.
"""

import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR_WEIGHTS = ROOT / "modelos" / "weights"
DIR_MANIFESTS = ROOT / "modelos" / "manifests"


def gerar_manifesto(tipo: str, extensao: str):
    """Gera manifest.txt com os checkpoints encontrados."""
    pasta_pesos = DIR_WEIGHTS / tipo
    if not pasta_pesos.is_dir() and tipo == "faster":
        pasta_alt = DIR_WEIGHTS / "faster r-cnn"
        if pasta_alt.is_dir():
            pasta_pesos = pasta_alt

    pasta_manifest = DIR_MANIFESTS / tipo
    pasta_manifest.mkdir(parents=True, exist_ok=True)
    manifest_file = pasta_manifest / "manifest.txt"

    checkpoints = sorted(pasta_pesos.glob(f"*{extensao}")) if pasta_pesos.is_dir() else []
    data_hora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(f"Manifesto {tipo.upper()} | {data_hora}\n")
        for ckpt in checkpoints:
            f.write(f"  - {ckpt.name} ({ckpt.stat().st_size / (1024*1024):.1f} MB)\n")

    print(f"Manifesto {tipo} atualizado: {manifest_file.relative_to(ROOT)}")


def main():
    gerar_manifesto("yolo", ".pt")
    gerar_manifesto("faster", ".pth")


if __name__ == "__main__":
    main()
