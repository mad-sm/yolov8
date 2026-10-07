"""Utilitas bersama: warna, titik acuan, penggambaran anotasi, penulis video.

Dipakai bareng oleh src/count.py (CLI) dan src/app.py (Streamlit) supaya
tampilan hasil keduanya identik dan logikanya tidak terduplikasi.

Tebal garis dan ukuran font diskalakan terhadap lebar frame (lihat scaled()),
supaya anotasi tetap terbaca di video CCTV beresolusi besar.
"""

import cv2

PALETTE = [
    (56, 168, 255), (56, 255, 148), (255, 128, 56), (255, 86, 180),
    (168, 86, 255), (255, 214, 56), (86, 235, 255), (120, 255, 86),
]

LINE_COLOR = (0, 235, 255)


def color_for(idx):
    return PALETTE[idx % len(PALETTE)]


def pick_device(requested="auto"):
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ref_point(box, mode="bottom"):
    """Titik acuan bounding box (x1,y1,x2,y2) untuk uji lintasan garis."""
    x1, y1, x2, y2 = box
    if mode == "center":
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    return ((x1 + x2) / 2, y2)  # bottom: titik roda, paling pas untuk kamera jalan


def open_writer(path, fps, size):
    """VideoWriter H.264 (bisa diputar di browser), fallback ke mp4v."""
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, fourcc
        w.release()
    raise RuntimeError(f"Tidak bisa membuat video writer untuk {path}")


def scaled(frame, base=1280.0):
    """Faktor skala tebal garis/font agar proporsional di frame besar (mis. 3450px)."""
    return max(1.0, frame.shape[1] / base)


def draw_box(frame, box, tid, cls_name, conf, color, point=None, trail=None):
    s = scaled(frame)
    th = max(1, int(2 * s))
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)

    label = f"#{tid} {cls_name} {conf:.2f}"
    fs = 0.45 * s
    (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(frame, (x1, y1 - tht - int(7 * s)), (x1 + tw + int(6 * s), y1), color, -1)
    cv2.putText(frame, label, (x1 + int(3 * s), y1 - int(5 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 15, 15), th, cv2.LINE_AA)

    if point is not None:
        cv2.circle(frame, (int(point[0]), int(point[1])), int(3 * s), color, -1)
    if trail:
        for a, b in zip(trail, list(trail)[1:]):
            cv2.line(frame, a, b, color, th)


def draw_line(frame, p1, p2):
    s = scaled(frame)
    a, b = tuple(map(int, p1)), tuple(map(int, p2))
    cv2.line(frame, a, b, LINE_COLOR, int(3 * s), cv2.LINE_AA)
    for p in (a, b):
        cv2.circle(frame, p, int(6 * s), LINE_COLOR, -1)


def draw_panel(frame, lines, origin=(12, 12), pad=10, alpha=0.55):
    """lines: list of (teks, warna BGR)."""
    s = scaled(frame)
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, max(1, int(s))
    pad = int(pad * s)
    gap = int(10 * s)

    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t, _ in lines]
    w = max(s_[0] for s_ in sizes) + pad * 2
    h = sum(s_[1] + gap for s_ in sizes) + pad * 2
    x, y = int(origin[0] * s), int(origin[1] * s)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (28, 28, 28), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), max(1, int(s)))

    cy = y + pad
    for (text, col), size in zip(lines, sizes):
        cy += size[1]
        cv2.putText(frame, text, (x + pad, cy), font, scale, col, th, cv2.LINE_AA)
        cy += gap
    return frame


def panel_lines(lc, names, per_frame_total, fps_disp=None):
    """Susun teks panel dari LineCounter."""
    pos, neg = lc.totals_per_direction()
    lines = [(f"TOTAL {lc.total}   {lc.label_pos} {pos}   {lc.label_neg} {neg}", (255, 255, 255))]
    sub = f"di layar: {per_frame_total}"
    if fps_disp is not None:
        sub += f"   fps: {fps_disp:.1f}"
    lines.append((sub, (185, 185, 185)))
    for cls_name in sorted(lc.counts):
        c = lc.counts[cls_name]
        cid = next((k for k, v in names.items() if v == cls_name), 0)
        lines.append((f"{cls_name:<12} {lc.label_pos} {c[lc.label_pos]:>4}  "
                      f"{lc.label_neg} {c[lc.label_neg]:>4}", color_for(cid)))
    return lines
