"""Unduh dataset 'Deteksi Kendaraan Indonesia 3' dari Roboflow.

Butuh API key Roboflow (Workspace Settings -> API Keys -> Private API Key).
Jalankan:
    export ROBOFLOW_API_KEY="xxxxx"
    python src/download_dataset.py

Format 'yolov11' dicoba lebih dulu, kalau SDK-nya belum kenal akan jatuh ke
'yolov8'. Keduanya sama saja isinya: susunan folder + label YOLO .txt.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORKSPACE = "puteri-marlisajasmine"
PROJECT = "deteksi-kendaraan-indonesia-3-ddnbr"
VERSION = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"))
    ap.add_argument("--workspace", default=WORKSPACE)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--version", type=int, default=VERSION)
    ap.add_argument("--format", default="yolov11", help="yolov11 / yolov8")
    ap.add_argument("--out", default=str(ROOT / "data" / "dataset"))
    args = ap.parse_args()

    if not args.api_key:
        sys.exit(
            "ROBOFLOW_API_KEY belum di-set.\n"
            "Ambil di https://app.roboflow.com -> Settings -> API Keys -> Private API Key\n"
            'lalu:  export ROBOFLOW_API_KEY="xxxxx"'
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    try:
        ds = version.download(args.format, location=args.out, overwrite=True)
    except Exception as e:  # format yolov11 tidak dikenal di SDK lama
        print(f"[!] format '{args.format}' gagal ({e}); fallback ke 'yolov8'")
        ds = version.download("yolov8", location=args.out, overwrite=True)

    data_yaml = Path(ds.location) / "data.yaml"
    print("\nDataset tersimpan di :", ds.location)
    print("data.yaml            :", data_yaml)
    if data_yaml.exists():
        print("\nIsi data.yaml:\n" + data_yaml.read_text())


if __name__ == "__main__":
    main()
