"""Training model deteksi kendaraan di dataset Roboflow (jalur lokal).

Contoh:
    python src/train.py --model yolo11s.pt --epochs 100

Di Mac (MPS) AMP dimatikan otomatis karena bikin NaN loss, dan workers=0.
Untuk training serius pakai GPU, gunakan notebook Colab di akar proyek:
colab_train_multi_model.ipynb (ganti satu baris MODEL untuk tiap varian).
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolo11s.pt",
                    help="yolo11n/s/m/l.pt — n=cepat, s=seimbang, m=akurat")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto (CUDA saja)")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | 0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30, help="early stopping")
    ap.add_argument("--name", default="kendaraan")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"data.yaml tidak ditemukan di {args.data}\n"
            "Jalankan dulu: python src/download_dataset.py"
        )

    from ultralytics import YOLO

    device = pick_device(args.device)
    # AMP di Apple MPS masih sering bikin NaN loss -> matikan.
    amp = device not in ("mps", "cpu")
    workers = 0 if device == "mps" else args.workers

    print(f"[i] device={device}  amp={amp}  workers={workers}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=workers,
        amp=amp,
        patience=args.patience,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        seed=0,
        plots=True,
        # augmentasi yang masuk akal untuk CCTV/dashcam jalan raya
        degrees=0.0,      # kamera jalan jarang miring
        fliplr=0.5,
        scale=0.5,        # kendaraan jauh vs dekat
        mosaic=1.0,
        close_mosaic=10,
    )

    metrics = model.val(data=args.data, device=device, split="test")
    print("\n=== Hasil evaluasi (test set) ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"\nBobot terbaik: {ROOT / 'runs' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
