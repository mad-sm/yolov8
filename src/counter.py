"""Logika penghitungan kendaraan berbasis garis (line crossing).

Sebuah track dihitung ketika titik acuannya (default: tengah-bawah bounding box)
berpindah sisi terhadap garis hitung A-B. Arah ditentukan dari tanda cross product,
sehingga satu garis bisa menghitung dua arah sekaligus.

Tiga pengaman terhadap salah hitung:
  - deadzone   : jitter kotak di sekitar garis tidak dihitung berulang
  - within_span: lintasan di perpanjangan garis (di luar segmen A-B) diabaikan
  - voting     : kelas diambil dari mayoritas prediksi sepanjang hidup track
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class CrossEvent:
    frame: int
    time_sec: float
    track_id: int
    cls_name: str
    direction: str


class LineCounter:
    """Penghitung lintasan garis untuk objek ber-track-id."""

    def __init__(self, p1, p2, dir_labels=("IN", "OUT"), deadzone=2.0, span_margin=0.05):
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.label_pos, self.label_neg = dir_labels
        self.deadzone = deadzone          # piksel; meredam jitter di sekitar garis
        self.span_margin = span_margin    # toleransi di luar ujung segmen

        self._len = max(
            ((self.p2[0] - self.p1[0]) ** 2 + (self.p2[1] - self.p1[1]) ** 2) ** 0.5, 1e-6
        )
        self.prev_side = {}                       # track_id -> -1 / +1
        self.cls_votes = defaultdict(Counter)     # track_id -> Counter(kelas)
        self.counts = defaultdict(lambda: {self.label_pos: 0, self.label_neg: 0})
        self.events: list[CrossEvent] = []
        self.seen_ids = set()

    # --- geometri -------------------------------------------------------
    def _signed_distance(self, pt):
        """Jarak bertanda titik ke garis (piksel). Tanda = sisi garis."""
        ax, ay = self.p1
        bx, by = self.p2
        cross = (bx - ax) * (pt[1] - ay) - (by - ay) * (pt[0] - ax)
        return cross / self._len

    def _within_span(self, pt):
        """True bila proyeksi titik jatuh di dalam rentang segmen A-B."""
        ax, ay = self.p1
        bx, by = self.p2
        t = ((pt[0] - ax) * (bx - ax) + (pt[1] - ay) * (by - ay)) / (self._len ** 2)
        return -self.span_margin <= t <= 1 + self.span_margin

    # --- update ---------------------------------------------------------
    def update(self, track_id, cls_name, point, frame_idx, time_sec):
        """Proses satu deteksi ter-track. Return CrossEvent bila melintas."""
        self.seen_ids.add(track_id)
        self.cls_votes[track_id][cls_name] += 1

        dist = self._signed_distance(point)
        if abs(dist) < self.deadzone:
            return None  # masih di zona mati, jangan ubah state

        side = 1 if dist > 0 else -1
        prev = self.prev_side.get(track_id)
        self.prev_side[track_id] = side

        if prev is None or prev == side:
            return None
        if not self._within_span(point):
            return None  # melintas di perpanjangan garis, bukan di segmennya

        direction = self.label_pos if side > 0 else self.label_neg
        stable_cls = self.cls_votes[track_id].most_common(1)[0][0]
        self.counts[stable_cls][direction] += 1

        ev = CrossEvent(frame_idx, round(time_sec, 3), track_id, stable_cls, direction)
        self.events.append(ev)
        return ev

    # --- ringkasan ------------------------------------------------------
    @property
    def total(self):
        return sum(v[self.label_pos] + v[self.label_neg] for v in self.counts.values())

    def totals_per_direction(self):
        pos = sum(v[self.label_pos] for v in self.counts.values())
        neg = sum(v[self.label_neg] for v in self.counts.values())
        return pos, neg

    def summary_rows(self):
        rows = []
        for cls_name in sorted(self.counts):
            c = self.counts[cls_name]
            rows.append(
                {
                    "kelas": cls_name,
                    self.label_pos: c[self.label_pos],
                    self.label_neg: c[self.label_neg],
                    "total": c[self.label_pos] + c[self.label_neg],
                }
            )
        return rows

    def forget(self, active_ids, keep=600):
        """Buang state track lama supaya memori tidak membengkak di video panjang."""
        if len(self.prev_side) <= keep:
            return
        for tid in list(self.prev_side):
            if tid not in active_ids:
                self.prev_side.pop(tid, None)
                self.cls_votes.pop(tid, None)


class TrackTrail:
    """Menyimpan jejak titik tiap track untuk digambar."""

    def __init__(self, maxlen=32):
        self.maxlen = maxlen
        self.trails = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, track_id, point):
        self.trails[track_id].append((int(point[0]), int(point[1])))
        return self.trails[track_id]

    def prune(self, active_ids):
        for tid in list(self.trails):
            if tid not in active_ids:
                del self.trails[tid]
