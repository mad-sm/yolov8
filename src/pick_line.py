"""Tentukan garis hitung dengan klik 2 titik di frame video.

    python src/pick_line.py --source data/videos/jalan.mov

Klik titik A, lalu titik B.  s = simpan, r = ulang, q = keluar.
Hasil disimpan ke data/line.json dan otomatis dipakai src/count.py maupun
aplikasi Streamlit.

Alternatif tanpa jendela OpenCV: atur garis lewat slider di src/app.py.
"""

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path video | 0 (webcam) | rtsp://...")
    ap.add_argument("--out", default=str(ROOT / "data" / "line.json"))
    ap.add_argument("--frame", type=int, default=0, help="ambil frame ke-N")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Tidak bisa membuka: {args.source}")
    if args.frame and not isinstance(src, int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Gagal membaca frame.")

    pts = []
    win = "Klik 2 titik garis hitung  |  s=simpan  r=ulang  q=keluar"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1280, frame.shape[1]), min(720, frame.shape[0]))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(canvas, p, 6, (0, 235, 255), -1)
            cv2.putText(canvas, "AB"[i], (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 235, 255), 2)
        if len(pts) == 2:
            cv2.line(canvas, pts[0], pts[1], (0, 235, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, "tekan 's' untuk simpan", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("q"):
            break
        elif k == ord("s") and len(pts) == 2:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"p1": list(pts[0]), "p2": list(pts[1])}, indent=2))
            print(f"Tersimpan: {out}  ->  p1={pts[0]} p2={pts[1]}")
            print(f'Atau langsung: --line "{pts[0][0]},{pts[0][1]},{pts[1][0]},{pts[1][1]}"')
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
