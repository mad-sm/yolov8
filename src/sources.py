"""Resolusi sumber video: file, webcam, RTSP, dan URL YouTube.

YouTube ditangani dengan yt-dlp:
  - video biasa  -> diunduh ke data/videos/ lalu diperlakukan sebagai file biasa
                    (bisa di-seek, jumlah frame diketahui, progress bar akurat)
  - siaran live  -> dipakai URL stream-nya langsung (tak berujung, jadi wajib
                    dibatasi jumlah frame oleh pemanggil)

Tidak butuh ffmpeg: format yang dipilih selalu satu berkas (video-only mp4),
jadi yt-dlp tidak perlu menggabung video+audio.
"""

import re
from pathlib import Path
from urllib.parse import urlparse

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"

YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
            "youtu.be", "www.youtu.be", "music.youtube.com"}


def is_youtube(src) -> bool:
    if not isinstance(src, str):
        return False
    if src.startswith("ytsearch"):
        return True
    try:
        return (urlparse(src).hostname or "").lower() in YT_HOSTS
    except ValueError:
        return False


def _fmt(max_height: int) -> str:
    """Satu berkas saja (tanpa merge), prioritas H.264 agar OpenCV senang."""
    return (f"bv*[height<={max_height}][vcodec^=avc1][ext=mp4]/"
            f"bv*[height<={max_height}][ext=mp4]/"
            f"bv*[height<={max_height}]/b[height<={max_height}]/b")


def _safe_name(title: str) -> str:
    name = re.sub(r"[^\w\s-]", "", title).strip()
    return re.sub(r"[\s]+", "_", name)[:60] or "youtube"


def youtube_info(url: str, max_height: int = 1080) -> dict:
    """Metadata video tanpa mengunduh."""
    from yt_dlp import YoutubeDL

    with YoutubeDL({"quiet": True, "no_warnings": True, "format": _fmt(max_height),
                    "socket_timeout": 20}) as ydl:
        info = ydl.extract_info(url, download=False)
    while isinstance(info, dict) and info.get("entries"):
        info = info["entries"][0]
    return {
        "id": info.get("id"),
        "title": info.get("title") or "youtube",
        "duration": info.get("duration"),
        "is_live": bool(info.get("is_live")),
        "width": info.get("width"),
        "height": info.get("height"),
        "url": info.get("url"),
        "webpage_url": info.get("webpage_url", url),
    }


def youtube_download(url: str, max_height: int = 1080, out_dir: Path = None,
                     progress_hook=None) -> Path:
    """Unduh video YouTube ke out_dir. Kalau sudah ada, pakai yang lama."""
    from yt_dlp import YoutubeDL

    out_dir = Path(out_dir or VIDEO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = youtube_info(url, max_height)
    if info["is_live"]:
        raise ValueError("Ini siaran live — tidak bisa diunduh. "
                         "Pakai mode stream dengan batas jumlah frame.")

    stem = f"yt_{info['id']}_{_safe_name(info['title'])}"
    for existing in out_dir.glob(f"{stem}.*"):
        if existing.stat().st_size > 0:
            return existing

    opts = {"quiet": True, "no_warnings": True, "format": _fmt(max_height),
            "outtmpl": str(out_dir / f"{stem}.%(ext)s"), "socket_timeout": 20}
    if progress_hook:
        opts["progress_hooks"] = [progress_hook]

    with YoutubeDL(opts) as ydl:
        res = ydl.extract_info(info["webpage_url"], download=True)
    while isinstance(res, dict) and res.get("entries"):
        res = res["entries"][0]

    path = Path(res.get("requested_downloads", [{}])[0].get("filepath")
                or ydl.prepare_filename(res))
    if not path.exists():
        cands = sorted(out_dir.glob(f"{stem}.*"))
        if not cands:
            raise FileNotFoundError(f"Unduhan gagal: {stem}")
        path = cands[0]
    return path


def resolve_source(src, yt_mode: str = "auto", max_height: int = 1080,
                   progress_hook=None):
    """Kembalikan (sumber_untuk_opencv, info).

    yt_mode: 'auto'   -> live pakai stream, selain itu diunduh
             'stream' -> selalu pakai URL stream (cepat, tapi URL kedaluwarsa ~6 jam)
             'download' -> selalu unduh
    """
    if isinstance(src, str) and src.isdigit():
        return int(src), {"jenis": "webcam", "nama": f"cam{src}"}

    if not is_youtube(src):
        return src, {"jenis": "file", "nama": Path(str(src)).stem}

    info = youtube_info(src, max_height)
    live = info["is_live"]

    if yt_mode == "stream" or (yt_mode == "auto" and live):
        return info["url"], {"jenis": "youtube-live" if live else "youtube-stream",
                             "nama": _safe_name(info["title"]), "live": live,
                             "judul": info["title"], "durasi": info["duration"]}

    path = youtube_download(src, max_height, progress_hook=progress_hook)
    return str(path), {"jenis": "youtube-file", "nama": path.stem, "live": False,
                       "judul": info["title"], "durasi": info["duration"],
                       "path": str(path)}


class LiveFrameReader:
    """Pembaca frame untuk siaran live (YouTube/RTSP).

    Kenapa perlu kelas sendiri: siaran live dikirim per segmen HLS. Kalau frame
    dibaca berurutan seperti file, pembacaan menumpuk di belakang tepi siaran —
    satu segmen habis dilahap cepat, lalu menunggu segmen berikutnya, dan karena
    inferensi lebih lambat dari 30 fps, jaraknya makin jauh sampai segmen lama
    hilang dari playlist dan stream dianggap putus.

    Solusinya: satu thread membaca terus-menerus supaya selalu menempel di tepi
    siaran, menampung frame di antrean pendek (maxlen). Selama mesin sanggup,
    frame diproses BERURUTAN - penting karena ByteTrack mencocokkan ID antar
    frame berdekatan. Kalau mesin tertinggal, antrean penuh dan frame terlama
    dibuang otomatis, jadi jeda ke tepi siaran tetap terbatas.

    maxlen dipilih ~2 detik (64 frame @30fps). Segmen HLS diunduh dan didekode
    jauh lebih cepat dari real-time, jadi frame datang berombak: ~140 frame
    sekaligus, lalu diam beberapa detik. Antrean yang terlalu pendek membuang
    sebagian besar ombak itu percuma. Antrean yang terlalu panjang memakan RAM
    (64 frame 720p ~ 180 MB) dan menambah jeda dari waktu nyata.
    """

    def __init__(self, url, max_height=1080, reopen_after=3, maxlen=64):
        import threading
        from collections import deque

        self.url = url
        self.max_height = max_height
        self.reopen_after = reopen_after
        self.reopen_count = 0
        self.dropped = 0

        self._lock = threading.Lock()
        self._q = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._cap = None

        if not self._open():
            raise RuntimeError(f"Tidak bisa membuka siaran: {url}")
        ok, first = self._read_blocking(timeout=30)
        if not ok:
            raise RuntimeError("Siaran terbuka tapi tidak mengirim frame.")
        self.height, self.width = first.shape[:2]
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # --- internal ---
    def _stream_url(self):
        if is_youtube(self.url):
            return youtube_info(self.url, self.max_height)["url"]
        return self.url

    def _open(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._stream_url())
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return self._cap.isOpened()

    def _read_blocking(self, timeout=30):
        import time as _t

        t0 = _t.time()
        while _t.time() - t0 < timeout:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._q.append(frame)
                return True, frame
            _t.sleep(0.2)
        return False, None

    def _loop(self):
        import time as _t

        fails = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                fails += 1
                if fails >= self.reopen_after:
                    self.reopen_count += 1
                    self._open()
                    fails = 0
                _t.sleep(0.3)
                continue
            fails = 0
            with self._lock:
                if len(self._q) == self._q.maxlen:
                    self.dropped += 1   # antrean penuh: frame terlama terbuang
                self._q.append(frame)

    # --- publik ---
    def read(self, timeout=30.0):
        """Ambil frame terlama di antrean (urutan waktu terjaga)."""
        import time as _t

        t0 = _t.time()
        while _t.time() - t0 < timeout:
            with self._lock:
                if self._q:
                    return True, self._q.popleft()
            _t.sleep(0.005)
        return False, None

    @property
    def backlog(self):
        with self._lock:
            return len(self._q)

    def release(self):
        self._stop.set()
        if self._cap is not None:
            self._cap.release()
