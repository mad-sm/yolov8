"""Aplikasi Streamlit: deteksi & hitung kendaraan di jalan raya.

Jalankan:
    cd ~/vehicle-counter && source .venv/bin/activate
    streamlit run src/app.py

Antarmuka untuk seluruh fitur CLI: pilih bobot, sumber video (upload / folder /
URL YouTube / webcam / RTSP), atur garis hitung lewat slider dengan pratinjau,
lalu jalankan. Hasil tampil sebagai metrik langsung, tabel per kelas, grafik
volume per menit, video beranotasi, dan unduhan CSV.

Daftar kelas dibaca langsung dari bobot yang dipilih, jadi kalau nanti dataset
ditambah kelas baru (mis. pickup) aplikasi ini tidak perlu diubah.
"""

import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources
from counter import LineCounter, TrackTrail
from viz import (color_for, draw_box, draw_line, draw_panel, open_writer,
                 panel_lines, pick_device, ref_point)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

st.set_page_config(page_title="Hitung Kendaraan", layout="wide")


# --------------------------------------------------------------------------
# helper
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model(weights_path: str):
    from ultralytics import YOLO

    m = YOLO(weights_path)
    return m, dict(m.names)


@st.cache_data(show_spinner=False)
def read_frame(video_path: str, pos: float = 0.3):
    """Ambil satu frame untuk pratinjau garis."""
    cap = cv2.VideoCapture(video_path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * pos))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, None
    return frame, {"n": int(n), "fps": fps, "w": frame.shape[1], "h": frame.shape[0]}


@st.cache_data(show_spinner=False, ttl=300)
def read_live_frame(stream_url: str):
    """Satu frame dari siaran live, untuk pratinjau penempatan garis."""
    cap = cv2.VideoCapture(stream_url)
    frame = None
    for _ in range(15):          # buang frame awal yang kadang rusak
        ok, f = cap.read()
        if ok:
            frame = f
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    if frame is None:
        return None, None
    return frame, {"n": 0, "fps": fps, "w": frame.shape[1], "h": frame.shape[0]}


def bersih(teks: str) -> str:
    """Buang emoji dari judul YouTube supaya tampilan tetap polos."""
    import re
    return re.sub(r"[^\w\s\-.,:;()\[\]/&'\"]+", "", teks).strip()


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_weights():
    """Bobot yang tersedia, terbaru dulu - sama urutannya dengan src/count.py."""
    runs = sorted(ROOT.glob("runs/**/weights/*.pt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in dict.fromkeys(runs + sorted(ROOT.glob("*.pt")))]


def rate_per_hour(n, seconds):
    return n * 3600 / seconds if seconds > 0 else 0


# --------------------------------------------------------------------------
# sidebar: sumber & model
# --------------------------------------------------------------------------
st.sidebar.title("Pengaturan")

weights_list = find_weights()
if not weights_list:
    st.error("Tidak ada file bobot (.pt) ditemukan. Training dulu, atau taruh best.pt "
             "di runs/kendaraan/weights/.")
    st.stop()

weights = st.sidebar.selectbox(
    "Bobot model", weights_list,
    index=next((i for i, w in enumerate(weights_list) if "best" in w), 0),
    format_func=lambda p: str(Path(p).relative_to(ROOT)),
    help="Terurut dari yang terbaru. Pastikan ini model yang kamu maksud.",
)

model, NAMES = load_model(weights)
st.sidebar.caption(f"Kelas terdeteksi: {', '.join(NAMES.values())}")

st.sidebar.divider()
src_mode = st.sidebar.radio(
    "Sumber video", ["Upload file", "File di folder", "URL YouTube", "Webcam / RTSP"])

video_path = None
if src_mode == "Upload file":
    up = st.sidebar.file_uploader("Pilih video", type=["mp4", "mov", "avi", "mkv"])
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / f"upload_{up.name}"
        tmp.write_bytes(up.getbuffer())
        video_path = str(tmp)
elif src_mode == "File di folder":
    vids = sorted(str(p) for ext in ("mp4", "mov", "avi", "mkv")
                  for p in (ROOT / "data" / "videos").glob(f"*.{ext}"))
    if vids:
        video_path = st.sidebar.selectbox("Video", vids,
                                          format_func=lambda p: Path(p).name)
    else:
        st.sidebar.warning("data/videos/ kosong")
elif src_mode == "URL YouTube":
    yt_url = st.sidebar.text_input("URL YouTube",
                                   placeholder="https://www.youtube.com/watch?v=...")
    yt_h = st.sidebar.select_slider("Resolusi maks", [480, 720, 1080], value=720)
    if st.sidebar.button("Ambil video", use_container_width=True):
        if not yt_url.strip():
            st.sidebar.warning("URL-nya belum diisi.")
        else:
            try:
                with st.spinner("Membaca info video..."):
                    info = sources.youtube_info(yt_url, yt_h)
                if info["is_live"]:
                    st.session_state["yt"] = {"path": info["url"], "live": True,
                                              "judul": info["title"]}
                else:
                    with st.spinner(f"Mengunduh: {info['title'][:40]}..."):
                        path = sources.youtube_download(yt_url, yt_h)
                    st.session_state["yt"] = {"path": str(path), "live": False,
                                              "judul": info["title"]}
                read_frame.clear()
            except Exception as e:
                st.session_state.pop("yt", None)
                st.sidebar.error(f"Gagal: {e}")
    if "yt" in st.session_state:
        yt = st.session_state["yt"]
        video_path = yt["path"]
        tag = "LIVE" if yt["live"] else "terunduh"
        st.sidebar.caption(f"{tag} · {bersih(yt['judul'])[:45]}")
else:
    video_path = st.sidebar.text_input("Sumber", "0",
                                       help="0 = webcam, atau rtsp://user:pass@ip:554/stream")

st.sidebar.divider()
st.sidebar.subheader("Deteksi")
conf = st.sidebar.slider("Ambang keyakinan (conf)", 0.05, 0.90, 0.30, 0.05)
imgsz = st.sidebar.select_slider("Ukuran inferensi", [640, 960, 1280, 1600, 1920], value=1280,
                                 help="Naikkan untuk kendaraan kecil/jauh, tapi lebih lambat")
sel_classes = st.sidebar.multiselect("Kelas yang dihitung", list(NAMES.values()),
                                     default=list(NAMES.values()))
max_area = st.sidebar.slider("Buang kotak > % luas frame", 0, 100, 0, 1,
                             help="0 = mati. Berguna menyaring kotak raksasa salah deteksi.")
vid_stride = st.sidebar.select_slider("Proses tiap N frame", [1, 2, 3, 5], value=1)

st.sidebar.subheader("Tracking (ByteTrack)")
track_buffer = st.sidebar.slider("track_buffer", 15, 150, 45, 5,
                                 help="Berapa frame ID dipertahankan saat objek tertutup")
ref_mode = st.sidebar.radio("Titik acuan", ["bottom", "center"], horizontal=True,
                            help="bottom = titik roda, paling pas untuk kamera jalan")


# --------------------------------------------------------------------------
# halaman utama
# --------------------------------------------------------------------------
st.title("Deteksi & Penghitungan Kendaraan")
st.caption("YOLO11 → ByteTrack → penghitungan lintasan garis")

if not video_path:
    if src_mode == "URL YouTube":
        st.info("Tempel URL YouTube di panel kiri lalu tekan **Ambil video**.")
    else:
        st.info("Pilih sumber video di panel kiri untuk mulai.")
    st.stop()

yt_live = src_mode == "URL YouTube" and st.session_state.get("yt", {}).get("live")
is_stream = (src_mode == "Webcam / RTSP") or yt_live

if yt_live:
    with st.spinner("Mengambil frame dari siaran untuk pratinjau..."):
        frame0, meta = read_live_frame(video_path)
elif is_stream:
    frame0, meta = None, None
else:
    frame0, meta = read_frame(video_path)

if frame0 is None and not is_stream:
    st.error(f"Tidak bisa membaca video: {video_path}")
    st.stop()
if yt_live and frame0 is None:
    st.error("Siaran terbuka tapi tidak mengirim frame. Coba ambil ulang videonya.")
    st.stop()

# ---- pengaturan garis ----
st.subheader("1. Garis hitung")

if frame0 is None:
    st.warning("Untuk webcam/RTSP, pratinjau garis tidak tersedia. "
               "Masukkan ukuran frame secara manual.")
    W = st.number_input("Lebar frame", 320, 8000, 1920)
    H = st.number_input("Tinggi frame", 240, 8000, 1080)
elif yt_live:
    W, H = meta["w"], meta["h"]
    st.caption(f"SIARAN LANGSUNG · {W}×{H} px · {meta['fps']:.1f} fps. "
               "Pratinjau di bawah diambil dari siaran barusan.")
else:
    W, H = meta["w"], meta["h"]
    st.caption(f"{W}×{H} px · {meta['fps']:.1f} fps · {meta['n']} frame "
               f"(~{meta['n'] / meta['fps']:.1f} detik)")

line_file = ROOT / "data" / "line.json"
default = {"p1": [int(W * 0.05), int(H * 0.6)], "p2": [int(W * 0.95), int(H * 0.6)]}
if line_file.exists():
    try:
        saved = json.loads(line_file.read_text())
        # Keempat koordinat harus muat di frame ini. Kalau tidak (mis. garis
        # disimpan untuk video beresolusi lain), pakai default - jangan diumpankan
        # ke slider karena nilai di luar rentang membuat Streamlit error.
        if all(0 <= saved[k][i] <= lim
               for k in ("p1", "p2")
               for i, lim in ((0, W), (1, H))):
            default = saved
        else:
            st.info("Garis tersimpan di data/line.json dibuat untuk video beresolusi "
                    "lain, jadi tidak dipakai. Atur garis baru di bawah lalu simpan.")
    except Exception:
        pass

c1, c2 = st.columns(2)
with c1:
    x1 = st.slider("Titik A · x", 0, W, int(default["p1"][0]))
    y1 = st.slider("Titik A · y", 0, H, int(default["p1"][1]))
with c2:
    x2 = st.slider("Titik B · x", 0, W, int(default["p2"][0]))
    y2 = st.slider("Titik B · y", 0, H, int(default["p2"][1]))

lc1, lc2 = st.columns(2)
lbl_pos = lc1.text_input("Label arah 1", "MASUK")
lbl_neg = lc2.text_input("Label arah 2", "KELUAR")

if frame0 is not None:
    prev = frame0.copy()
    draw_line(prev, (x1, y1), (x2, y2))
    st.image(bgr2rgb(prev), caption="Pratinjau garis hitung", use_container_width=True)
    st.caption("Taruh garis melintang jalan di area kendaraan terlihat penuh dan tidak "
               "saling menumpuk. Hindari tepi frame — ID belum stabil di sana.")

if st.button("Simpan garis ke data/line.json"):
    line_file.parent.mkdir(parents=True, exist_ok=True)
    line_file.write_text(json.dumps({"p1": [x1, y1], "p2": [x2, y2]}, indent=2))
    st.success(f"Tersimpan: A=({x1},{y1}) B=({x2},{y2})")

# ---- jalankan ----
st.subheader("2. Jalankan penghitungan")

opt1, opt2 = st.columns(2)
save_video = opt1.checkbox("Simpan video hasil", value=True)
show_live = opt2.checkbox("Tampilkan proses langsung", value=True)

max_frames = None
if is_stream:
    max_frames = st.number_input("Berhenti setelah N frame", 100, 100000, 1000, 100,
                                 help="Stream tidak punya ujung; batasi agar bisa berhenti.")

if st.button("Mulai", type="primary", use_container_width=True):
    device = pick_device("auto")
    class_filter = [k for k, v in NAMES.items() if v in sel_classes] or None

    tracker_cfg = ROOT / "configs" / "bytetrack.yaml"
    if tracker_cfg.exists():
        base = tracker_cfg.read_text()
        tmp_cfg = Path(tempfile.gettempdir()) / "bytetrack_app.yaml"
        out_lines = []
        for ln in base.splitlines():
            if ln.strip().startswith("track_buffer:"):
                ln = f"track_buffer: {track_buffer}"
            out_lines.append(ln)
        tmp_cfg.write_text("\n".join(out_lines))
        tracker = str(tmp_cfg)
    else:
        tracker = "bytetrack.yaml"

    source = int(video_path) if str(video_path).isdigit() else video_path
    fps_in = (meta["fps"] if meta else 25.0)
    total_frames = 0 if yt_live else ((meta["n"] // vid_stride if meta else max_frames) or 0)
    if yt_live:
        total_frames = max_frames or 0

    lc = LineCounter((x1, y1), (x2, y2), dir_labels=(lbl_pos, lbl_neg))
    trails = TrackTrail()

    writer = None
    vid_out = OUT / "app_counted.mp4"
    if save_video:
        writer, _ = open_writer(vid_out, fps_in / vid_stride, (W, H))

    prog = st.progress(0.0, "Menyiapkan...")
    metric_box = st.container()
    m1, m2, m3, m4 = metric_box.columns(4)
    ph_total, ph_pos, ph_neg, ph_fps = m1.empty(), m2.empty(), m3.empty(), m4.empty()
    ph_img = st.empty()

    track_kw = dict(tracker=tracker, persist=True, conf=conf, imgsz=imgsz,
                    classes=class_filter, device=device, verbose=False)
    reader = None

    if yt_live:
        # Siaran live perlu pembaca sendiri: Ultralytics membaca berurutan dan
        # cepat tertinggal dari tepi siaran (lihat sources.LiveFrameReader).
        reader = sources.LiveFrameReader(st.session_state["yt"]["path"])

        def live_results():
            while True:
                ok, frame_in = reader.read(timeout=30)
                if not ok:
                    return
                yield model.track(frame_in, **track_kw)[0]

        results = live_results()
    else:
        results = model.track(source=source, stream=True,
                              vid_stride=vid_stride, **track_kw)

    frame_idx, n_dropped = 0, 0
    fps_disp, t_prev, t_start = 0.0, time.time(), time.time()
    events = []

    try:
        for res in results:
            frame = res.orig_img.copy()
            frame_idx += 1
            t_sec = frame_idx * vid_stride / fps_in

            boxes = res.boxes
            active_ids, per_frame = set(), Counter()

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                clss = boxes.cls.int().cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                farea = frame.shape[0] * frame.shape[1]

                for box, tid, ci, cf in zip(xyxy, ids, clss, confs):
                    tid, cname = int(tid), NAMES[int(ci)]
                    if max_area > 0:
                        a = (box[2] - box[0]) * (box[3] - box[1]) / farea * 100
                        if a > max_area:
                            n_dropped += 1
                            continue

                    active_ids.add(tid)
                    per_frame[cname] += 1
                    pt = ref_point(box, ref_mode)
                    ev = lc.update(tid, cname, pt, frame_idx, t_sec)
                    if ev:
                        events.append({"frame": ev.frame, "detik": ev.time_sec,
                                       "track_id": ev.track_id, "kelas": ev.cls_name,
                                       "arah": ev.direction})
                    draw_box(frame, box, tid, cname, cf, color_for(int(ci)),
                             pt, trails.update(tid, pt))

            trails.prune(active_ids)
            lc.forget(active_ids)
            draw_line(frame, lc.p1, lc.p2)

            now = time.time()
            if now > t_prev:
                fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / (now - t_prev))
            t_prev = now
            draw_panel(frame, panel_lines(lc, NAMES, sum(per_frame.values()), fps_disp))

            if writer is not None:
                writer.write(frame)

            pos, neg = lc.totals_per_direction()
            ph_total.metric("Total melintas", lc.total)
            ph_pos.metric(lbl_pos, pos)
            ph_neg.metric(lbl_neg, neg)
            ph_fps.metric("fps proses", f"{fps_disp:.1f}")

            if show_live and frame_idx % 3 == 0:
                small = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
                ph_img.image(bgr2rgb(small), use_container_width=True)

            if total_frames:
                prog.progress(min(frame_idx / total_frames, 1.0),
                              f"Frame {frame_idx}/{total_frames}")
            else:
                prog.progress(0.5, f"Frame {frame_idx}")

            if max_frames and frame_idx >= max_frames:
                break
    finally:
        if reader is not None:
            reader.release()
        if writer is not None:
            writer.release()

    prog.progress(1.0, "Selesai")
    elapsed = time.time() - t_start
    dur = frame_idx * vid_stride / fps_in

    st.session_state["hasil"] = {
        "rows": lc.summary_rows(), "events": events, "total": lc.total,
        "pos": lc.totals_per_direction()[0], "neg": lc.totals_per_direction()[1],
        "unik": len(lc.seen_ids), "dropped": n_dropped, "frames": frame_idx,
        "durasi": dur, "elapsed": elapsed, "lbl": (lbl_pos, lbl_neg),
        "video": str(vid_out) if save_video else None,
    }

# ---- hasil ----
if "hasil" in st.session_state:
    h = st.session_state["hasil"]
    lbl_p, lbl_n = h["lbl"]

    st.subheader("3. Hasil")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total melintas", h["total"])
    k2.metric(lbl_p, h["pos"])
    k3.metric(lbl_n, h["neg"])
    k4.metric("Kendaraan unik terdeteksi", h["unik"])

    if h["durasi"] > 0:
        st.caption(
            f"Durasi terproses {h['durasi']:.1f} detik · laju "
            f"**{rate_per_hour(h['total'], h['durasi']):.0f} kendaraan/jam** "
            f"· {h['frames']} frame dalam {h['elapsed']:.0f} detik "
            f"({h['frames'] / max(h['elapsed'], 1e-6):.1f} fps)"
        )
    if h["dropped"]:
        st.caption(f"{h['dropped']} deteksi dibuang karena melebihi batas luas kotak.")

    if h["rows"]:
        df = pd.DataFrame(h["rows"])
        c1, c2 = st.columns([1, 1])
        c1.dataframe(df, use_container_width=True, hide_index=True)
        c2.bar_chart(df.set_index("kelas")[[lbl_p, lbl_n]])

        st.download_button("Ringkasan (CSV)", df.to_csv(index=False),
                           "ringkasan.csv", "text/csv")
    else:
        st.warning("Tidak ada kendaraan yang melintasi garis. "
                   "Cek posisi garis — mungkin di luar jalur kendaraan, "
                   "atau videonya terlalu pendek.")

    if h["events"]:
        ev = pd.DataFrame(h["events"])
        with st.expander(f"Rincian {len(ev)} lintasan"):
            st.dataframe(ev, use_container_width=True, hide_index=True)
            st.download_button("Event (CSV)", ev.to_csv(index=False),
                               "events.csv", "text/csv")

        if h["durasi"] > 10:
            ev["menit"] = (ev["detik"] // 60).astype(int)
            per_min = ev.groupby(["menit", "kelas"]).size().unstack(fill_value=0)
            st.caption("Volume per menit")
            st.bar_chart(per_min)

    if h["video"] and Path(h["video"]).exists():
        st.video(h["video"])
        st.download_button("Video hasil", Path(h["video"]).read_bytes(),
                           "hasil_hitung.mp4", "video/mp4")
