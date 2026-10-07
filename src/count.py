"""Deteksi + hitung kendaraan pada video / CCTV / webcam / YouTube.

Deteksi : YOLO (bobot hasil training dataset Roboflow)
Tracking: ByteTrack (bawaan Ultralytics, config di configs/bytetrack.yaml)
Hitung  : line crossing dua arah, per kelas kendaraan

Contoh:
    # file video
    python src/count.py --weights runs/kendaraan/weights/best.pt \
        --source data/videos/jalan.mov --line-file data/line.json --save --show

    # URL YouTube (otomatis diunduh dulu ke data/videos/)
    python src/count.py --source "https://www.youtube.com/watch?v=XXXX" --save

Kalau --weights tidak diberi, dipakai best.pt terbaru di runs/**/weights/.
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from counter import LineCounter, TrackTrail
from sources import LiveFrameReader, resolve_source
from viz import (color_for, draw_box, draw_line, draw_panel, open_writer,
                 panel_lines, pick_device, ref_point)

ROOT = Path(__file__).resolve().parent.parent




def load_line(args, frame_shape):
    if args.line:
        x1, y1, x2, y2 = [int(v) for v in args.line.split(",")]
        return (x1, y1), (x2, y2)
    if args.line_file and Path(args.line_file).exists():
        d = json.loads(Path(args.line_file).read_text())
        return tuple(d["p1"]), tuple(d["p2"])
    # default: garis horizontal di 60% tinggi frame
    h, w = frame_shape[:2]
    y = int(h * 0.6)
    print(f"[!] Garis hitung tidak diberikan, pakai default horizontal y={y}. "
          f"Bikin garis sendiri: python src/pick_line.py --source <video>")
    return (int(w * 0.05), y), (int(w * 0.95), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="path .pt; kalau tidak diberi, pakai best.pt terbaru di runs/")
    ap.add_argument("--source", required=True,
                    help="path video | 0 (webcam) | rtsp://... | URL YouTube")
    ap.add_argument("--yt-mode", default="auto", choices=["auto", "stream", "download"],
                    help="perlakuan URL YouTube: auto (live=stream, selain itu unduh), "
                         "stream (jangan unduh), download (selalu unduh)")
    ap.add_argument("--yt-height", type=int, default=1080,
                    help="resolusi maksimum yang diambil dari YouTube")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="berhenti setelah N frame (wajib untuk siaran live)")
    ap.add_argument("--line", help="x1,y1,x2,y2 (piksel)")
    ap.add_argument("--line-file", default=str(ROOT / "data" / "line.json"))
    ap.add_argument("--labels", default="MASUK,KELUAR", help="label 2 arah, dipisah koma")
    ap.add_argument("--ref", default="bottom", choices=["bottom", "center"],
                    help="titik acuan bbox untuk uji lintasan")
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "bytetrack.yaml"))
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--classes", default="", help="filter kelas, mis. 'mobil,bus' atau '0,2'")
    ap.add_argument("--max-area", type=float, default=0.0,
                    help="buang deteksi yang luasnya > N%% luas frame (0 = mati). "
                         "Ampuh untuk menyaring kotak raksasa hasil salah deteksi.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vid-stride", type=int, default=1, help="proses tiap N frame (percepat)")
    ap.add_argument("--show", action="store_true", help="tampilkan window (q = keluar)")
    ap.add_argument("--save", action="store_true", help="simpan video hasil anotasi")
    ap.add_argument("--out-dir", default=str(ROOT / "output"))
    ap.add_argument("--no-trail", action="store_true")
    args = ap.parse_args()

    if args.weights is None:
        found = sorted(ROOT.glob("runs/**/weights/best.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            raise SystemExit(
                "Tidak ada bobot di runs/**/weights/best.pt.\n"
                "Train dulu, atau tunjuk --weights ke file .pt."
            )
        args.weights = str(found[0])
        others = [p.parent.parent.name for p in found[1:]]
        print(f"[i] --weights tidak diberi, pakai yang terbaru: "
              f"{Path(args.weights).relative_to(ROOT)}")
        if others:
            print(f"    (ada juga: {', '.join(others)} — pilih dengan --weights)")
    elif not Path(args.weights).exists():
        raise SystemExit(
            f"Bobot tidak ditemukan: {args.weights}\n"
            "Train dulu (python src/train.py) atau tunjuk --weights ke file .pt lain."
        )

    from ultralytics import YOLO

    device = pick_device(args.device)
    model = YOLO(args.weights)
    names = model.names
    print(f"[i] device={device}  kelas={list(names.values())}")

    class_filter = None
    if args.classes:
        wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
        name2id = {v.lower(): k for k, v in names.items()}
        class_filter = [int(c) if c.isdigit() else name2id[c.lower()] for c in wanted]
        print(f"[i] filter kelas -> {[names[c] for c in class_filter]}")

    tracker = args.tracker if Path(args.tracker).exists() else "bytetrack.yaml"

    source, src_info = resolve_source(args.source, yt_mode=args.yt_mode,
                                      max_height=args.yt_height)
    if src_info["jenis"].startswith("youtube"):
        print(f"[i] YouTube: {src_info.get('judul','')} ({src_info['jenis']})")
    if src_info.get("live") and not args.max_frames:
        raise SystemExit("Siaran live tidak berujung — beri --max-frames, mis. --max-frames 3000")
    live = bool(src_info.get("live"))
    reader = None
    if live:
        print("[i] mode siaran langsung: frame basi dibuang agar tidak tertinggal "
              "dari tepi siaran")
        reader = LiveFrameReader(args.source, max_height=args.yt_height)
        W, H, fps_in = reader.width, reader.height, reader.fps
        first = None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Tidak bisa membuka source: {args.source}")
        ok, first = cap.read()
        if not ok:
            raise SystemExit("Gagal membaca frame pertama.")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        H, W = first.shape[:2]
        cap.release()

    p1, p2 = load_line(args, (H, W))
    dir_labels = tuple(l.strip() for l in args.labels.split(",")[:2])
    lc = LineCounter(p1, p2, dir_labels=dir_labels)
    trails = TrackTrail()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src_info["nama"]
    writer = None
    if args.save:
        vid_path = out_dir / f"{stem}_counted.mp4"
        writer, fourcc = open_writer(vid_path, fps_in / args.vid_stride, (W, H))
        print(f"[i] video hasil -> {vid_path}  (codec {fourcc})")

    events_path = out_dir / f"{stem}_events.csv"
    ev_file = open(events_path, "w", newline="")
    ev_writer = csv.writer(ev_file)
    ev_writer.writerow(["frame", "detik", "track_id", "kelas", "arah"])

    track_kw = dict(tracker=tracker, persist=True, conf=args.conf, iou=args.iou,
                    imgsz=args.imgsz, classes=class_filter, device=device,
                    verbose=False)

    if live:
        def live_results():
            """Ambil frame terbaru dari siaran, lacak satu per satu."""
            while True:
                ok, frame_in = reader.read(timeout=30)
                if not ok:
                    print("[!] siaran berhenti mengirim frame selama 30 detik")
                    return
                yield model.track(frame_in, **track_kw)[0]

        results = live_results()
    else:
        results = model.track(source=source, stream=True,
                              vid_stride=args.vid_stride, **track_kw)

    frame_idx = 0
    n_dropped = 0
    t_start = time.time()
    fps_disp = 0.0
    t_prev = t_start

    try:
        for res in results:
            frame = res.orig_img.copy()
            frame_idx += 1
            t_sec = frame_idx * args.vid_stride / fps_in

            boxes = res.boxes
            active_ids, per_frame_cls = set(), Counter()

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                clss = boxes.cls.int().cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                frame_area = frame.shape[0] * frame.shape[1]
                for box, tid, ci, cf in zip(xyxy, ids, clss, confs):
                    tid = int(tid)
                    cls_name = names[int(ci)]

                    if args.max_area > 0:
                        area_pct = (box[2] - box[0]) * (box[3] - box[1]) / frame_area * 100
                        if area_pct > args.max_area:
                            n_dropped += 1
                            continue

                    active_ids.add(tid)
                    per_frame_cls[cls_name] += 1

                    pt = ref_point(box, args.ref)
                    ev = lc.update(tid, cls_name, pt, frame_idx, t_sec)
                    if ev:
                        ev_writer.writerow(
                            [ev.frame, ev.time_sec, ev.track_id, ev.cls_name, ev.direction]
                        )

                    trail = None if args.no_trail else trails.update(tid, pt)
                    draw_box(frame, box, tid, cls_name, cf, color_for(int(ci)), pt, trail)

            trails.prune(active_ids)
            lc.forget(active_ids)
            draw_line(frame, lc.p1, lc.p2)

            now = time.time()
            if now - t_prev > 0:
                fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / (now - t_prev))
            t_prev = now

            draw_panel(frame, panel_lines(lc, names, sum(per_frame_cls.values()), fps_disp))

            if writer is not None:
                writer.write(frame)
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[i] berhenti di batas {args.max_frames} frame")
                break
            if args.show:
                cv2.imshow("Deteksi & Hitung Kendaraan", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[i] dihentikan user")
                    break
    except KeyboardInterrupt:
        print("\n[i] dihentikan (Ctrl-C)")
    finally:
        if reader is not None:
            reader.release()
        ev_file.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # ---- ringkasan ----
    rows = lc.summary_rows()
    summary_path = out_dir / f"{stem}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kelas", dir_labels[0], dir_labels[1], "total"])
        w.writeheader()
        w.writerows(rows)

    elapsed = time.time() - t_start
    pos, neg = lc.totals_per_direction()
    print("\n=== RINGKASAN ===")
    print(f"{'kelas':<14}{dir_labels[0]:>8}{dir_labels[1]:>8}{'total':>8}")
    for r in rows:
        print(f"{r['kelas']:<14}{r[dir_labels[0]]:>8}{r[dir_labels[1]]:>8}{r['total']:>8}")
    print("-" * 38)
    print(f"{'SEMUA':<14}{pos:>8}{neg:>8}{lc.total:>8}")
    print(f"\nkendaraan unik terdeteksi : {len(lc.seen_ids)}")
    if reader is not None:
        print(f"frame siaran dilewati     : {reader.dropped} "
              f"(antrean penuh; 0 = tidak pernah tertinggal)")
        if reader.reopen_count:
            print(f"sambung ulang siaran      : {reader.reopen_count}x")
    if args.max_area > 0:
        print(f"deteksi dibuang (>{args.max_area:g}% frame) : {n_dropped}")
    print(f"frame diproses            : {frame_idx}  ({elapsed:.1f}s, "
          f"{frame_idx / max(elapsed, 1e-6):.1f} fps)")
    print(f"event  -> {events_path}")
    print(f"ringkasan -> {summary_path}")


if __name__ == "__main__":
    main()
