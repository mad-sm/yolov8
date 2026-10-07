# Deteksi & Hitung Kendaraan di Jalan Raya

Pipeline: **YOLOv8 (deteksi) → ByteTrack (tracking) → line crossing (hitung)**.
Dataset: [Deteksi Kendaraan Indonesia 3 — Roboflow](https://app.roboflow.com/puteri-marlisajasmine/deteksi-kendaraan-indonesia-3-ddnbr/3) (versi 3).

Notebook training yang tersedia:

| Notebook | Model | Catatan |
|---|---|---|
| `colab_train_multi_model.ipynb` | bebas | ganti satu baris `MODEL` untuk tiap varian |
| `colab_train_yolov8n.ipynb` | YOLOv8n | ringan, banyak pembanding di literatur |
| `colab_train_kendaraan.ipynb` | YOLO11s | run pertama, dipakai sebagai pembanding |

Semuanya identik selain modelnya, jadi angkanya bisa langsung dibandingkan — lihat
[Catatan kualitas model](#catatan-kualitas-model) untuk hasil YOLOv8s vs YOLO11s.

Kenapa perlu tracking, bukan sekadar deteksi: deteksi saja menghitung ulang kendaraan
yang sama di tiap frame. ByteTrack memberi **ID unik** per kendaraan, jadi satu mobil
dihitung sekali saat ID-nya melintasi garis hitung.

## Struktur

```
vehicle-counter/
├── configs/bytetrack.yaml   # parameter ByteTrack (bisa ditune)
├── src/
│   ├── app.py               # APLIKASI STREAMLIT
│   ├── sources.py           # resolusi sumber: file / webcam / RTSP / YouTube
│   ├── viz.py               # warna, gambar kotak/garis/panel (dipakai CLI + app)
│   ├── download_dataset.py  # tarik dataset dari Roboflow
│   ├── train.py             # training YOLOv8
│   ├── pick_line.py         # klik 2 titik untuk garis hitung
│   ├── counter.py           # logika line crossing (dipakai count.py)
│   └── count.py             # PROGRAM UTAMA: deteksi + hitung
├── data/                    # dataset, video, line.json
├── runs/                    # hasil training
└── output/                  # video anotasi + CSV hasil hitung
```

## Aplikasi Streamlit

```bash
cd ~/vehicle-counter && source .venv/bin/activate
streamlit run src/app.py
```

Semua fitur CLI tersedia lewat antarmuka: pilih bobot, sumber video (upload / folder /
**URL YouTube** / webcam / RTSP), atur garis hitung dengan slider + pratinjau langsung,
lalu jalankan. Hasilnya tampil sebagai metrik langsung, tabel per kelas, grafik volume
per menit, video beranotasi, dan tombol unduh CSV.

Daftar kelas dibaca dari bobot yang dipilih, jadi kalau dataset ditambah kelas baru
(mis. `pickup`) aplikasi ini tidak perlu diubah sama sekali.

## Sumber YouTube

```bash
# video biasa - otomatis diunduh ke data/videos/ lalu diproses
python src/count.py --weights runs/kendaraan/weights/best.pt \
    --source "https://www.youtube.com/watch?v=XXXX" --yt-height 720 --save

# siaran live - wajib dibatasi jumlah frame
python src/count.py --weights runs/kendaraan/weights/best.pt \
    --source "https://www.youtube.com/watch?v=XXXX" --max-frames 3000 --save
```

| Flag | Guna |
|---|---|
| `--yt-mode` | `auto` (live=stream, selain itu unduh), `stream`, `download` |
| `--yt-height` | resolusi maksimum yang diambil (default 1080) |
| `--max-frames` | berhenti setelah N frame — **wajib** untuk siaran live |

Ditangani `yt-dlp` lewat `src/sources.py`. Format yang dipilih selalu satu berkas
(video-only mp4), jadi **tidak butuh ffmpeg**. Video yang sudah diunduh dipakai ulang,
tidak diunduh dua kali. URL stream YouTube kedaluwarsa ~6 jam, jadi untuk pemrosesan
panjang lebih aman mode unduh.

## Setup (sudah dikerjakan)

Venv Python 3.12 (arm64, jadi bisa pakai GPU `mps` M1) ada di `.venv/`:

```bash
cd ~/vehicle-counter
source .venv/bin/activate     # semua perintah di bawah asumsinya venv aktif
```

Kalau perlu install ulang: `uv pip install -r requirements.txt`.

## 1. Unduh dataset

Ambil **Private API Key** di Roboflow: Workspace → Settings → API Keys.

```bash
export ROBOFLOW_API_KEY="xxxxx"
python src/download_dataset.py
```

Hasil: `data/dataset/` berisi `train/ valid/ test/ data.yaml`.

## 2. Training

```bash
python src/train.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
```

- `yolov8n.pt` cepat tapi kurang akurat untuk kendaraan kecil/jauh; `yolov8s.pt` titik
  seimbang (ini default `--model`); `yolov8m.pt` paling akurat tapi berat di M1.
- Varian YOLO11 (`yolo11s.pt` dst) tetap bisa dipakai lewat `--model` — script dan
  pipeline hitungnya sama, jadi angkanya bisa langsung dibandingkan.
- Di M1 Pro, AMP dimatikan otomatis (bikin NaN loss di MPS) dan `workers=0`.
  Perkiraan: ~2–5 menit/epoch untuk ~2.000 gambar @640. Kalau kelamaan, sewa GPU di
  Colab/Kaggle dan jalankan script yang sama dengan `--device 0`.
- Bobot terbaik: `runs/kendaraan/weights/best.pt`. Script langsung evaluasi di test set
  dan cetak mAP50 / mAP50-95.

Target realistis untuk dataset CCTV jalan Indonesia: **mAP50 ≥ 0.80** sudah layak pakai.
Kalau di bawah itu, tambah epoch, naikkan `--imgsz` ke 960, atau perbaiki anotasi.

## 3. Tentukan garis hitung

```bash
python src/pick_line.py --source data/videos/jalan.mp4
```

Klik titik A dan B melintang jalan → tekan `s`. Tersimpan di `data/line.json`.

Tips penempatan: taruh garis di area **tengah frame** di mana kendaraan terlihat penuh
dan tidak saling menumpuk. Jangan di tepi frame (objek baru muncul, ID belum stabil).

## 4. Hitung kendaraan

```bash
python src/count.py \
  --weights runs/kendaraan_yolov8_v8/weights/best.pt \
  --source data/videos/jalan.mp4 \
  --line-file data/line.json \
  --save --show
```

Default-nya hitungannya **satu angka**: berapa kendaraan melewati garis. Kalau butuh
dipisah per arah, tambahkan `--two-way` (label bisa diganti lewat `--labels "MASUK,KELUAR"`):

```bash
python src/count.py --weights runs/kendaraan_yolov8_v8/weights/best.pt \
  --source data/videos/jalan.mp4 --two-way --labels "MASUK,KELUAR" --save
```

Di app Streamlit, mode ini ada sebagai checkbox **"Pisahkan hitungan per arah"**.

Kalau `--weights` tidak diisi, script otomatis ambil `best.pt` paling baru di
`runs/**/weights/`.

Sumber lain: `--source 0` (webcam), `--source rtsp://user:pass@ip:554/stream` (CCTV).

Output di `output/`:
| File | Isi |
|---|---|
| `<nama>_counted.mp4` | video dengan bbox, ID, jejak, garis, panel hitung |
| `<nama>_events.csv` | tiap lintasan: `frame, detik, track_id, kelas` (+ `arah` kalau `--two-way`) |
| `<nama>_summary.csv` | rekap per kelas: kolom `JUMLAH`, atau per arah + `total` kalau `--two-way` |

`events.csv` yang berisi timestamp itu bahan mentah untuk analisis lanjutan —
volume per jam, jam sibuk, komposisi jenis kendaraan.

### Opsi yang sering dipakai

| Flag | Guna |
|---|---|
| `--conf 0.3` | ambang deteksi; turunkan kalau banyak kendaraan terlewat |
| `--imgsz 960` | naikkan untuk kendaraan kecil/jauh (lebih lambat) |
| `--classes mobil,bus` | hitung kelas tertentu saja |
| `--vid-stride 2` | proses tiap 2 frame, ~2× lebih cepat (hitungan sedikit kurang akurat) |
| `--max-area 12` | buang deteksi yang luasnya >12% frame (kotak raksasa salah deteksi) |
| `--ref center` | titik acuan pakai tengah bbox, bukan tengah-bawah |
| `--no-trail` | matikan gambar jejak |
| `--two-way` | pisahkan hitungan per arah (default: satu angka total) |

## Catatan kualitas model

Model yang dipakai sekarang: **YOLOv8s**, `runs/kendaraan_yolov8_v8/weights/best.pt`
(100 epoch, imgsz 640). Kelas: `bus, mobil, motor, truk`.

Kedua bobot diukur ulang di **split `test`** yang sama (304 gambar, 1466 instance,
imgsz 640, 7 Okt 2026) supaya bisa dibandingkan apple-to-apple:

| Model | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| **YOLOv8s** (dipakai) | **0.945** | **0.730** | 0.927 | 0.887 |
| YOLO11s | 0.941 | 0.726 | 0.937 | 0.869 |

mAP50 per kelas (YOLOv8s): `truk` 0.966, `bus` 0.963, `mobil` 0.956, `motor` 0.895.
`motor` paling rendah di mAP50-95 (0.586) — objeknya kecil, jadi kotaknya kurang presisi
meski deteksinya kena.

Catatan: 1 gambar di split test dilewati karena anotasinya campur segment + detection
(`23978_truk_jpg.rf.f966bb01…`), jadi 304 dari 305 gambar yang terhitung.

### Uji di rekaman CCTV asli

Catatan di bawah ini berasal dari **run YOLO11s (19 Agu 2026)** di CCTV Simpang
Gondomanan (`data/videos/jalan.mov`, 3450×1942, 21 detik). **Belum diulang dengan bobot
YOLOv8s**, jadi belum diketahui apakah gejalanya masih sama:

| Kelas | Status |
|---|---|
| `motor` | bagus — kotak rapat, tracking stabil |
| `mobil` | bagus |
| `bus` | **rusak** — kotak membengkak, median 30% luas frame (mobil 1.8%, motor 0.95%) |
| `truk` | **tidak pernah muncul** |

Kendaraan besar (pickup) tetap terdeteksi, tapi diberi label `bus` dengan kotak raksasa —
jadi ini kegagalan lokalisasi, bukan sekadar salah kelas. Akibatnya kendaraan besar
belum bisa dihitung dengan andal.

**Sudah dicek:** mAP50 per kelas di split test ternyata tinggi untuk `bus` (0.963) dan
`truk` (0.966) — jadi hipotesis "dataset-nya rusak" **tidak terbukti**. Model bisa
mengenali bus/truk dengan baik di gambar yang sejenis data latihnya. Yang gagal adalah
generalisasi ke rekaman CCTV asli: resolusinya jauh lebih besar (3450×1942 vs 640),
sudutnya lebih tinggi, dan kendaraan besar muncul dalam skala yang tidak ada di dataset.
Ini domain gap, bukan anotasi busuk. Arah perbaikan: tambah sampel frame CCTV asli ke
dataset, atau naikkan `--imgsz` saat inferensi supaya skalanya lebih dekat.

**Untuk sekarang**, konfigurasi yang andal: `--classes motor,mobil`.

## Cara kerja penghitungan

`src/counter.py` menghitung **jarak bertanda** titik acuan kendaraan (default:
tengah-bawah bbox ≈ posisi roda di aspal) terhadap garis A–B. Saat tandanya berbalik,
kendaraan dianggap melintas. Tanda yang baru juga menunjukkan arah lintasan, jadi satu
garis bisa menghitung dua arah sekaligus — tapi pemisahan arah itu baru ditampilkan
kalau `--two-way` dinyalakan. Tiga pengaman:

- **deadzone 2 px** — meredam jitter bbox di sekitar garis agar tidak dihitung ganda.
- **cek rentang segmen** — lintasan di perpanjangan garis (di luar A–B) diabaikan.
- **voting kelas per track** — label kelas diambil dari mayoritas prediksi sepanjang
  hidup track, jadi mobil yang sesekali salah terbaca "truk" tetap tercatat benar.

## Tuning ByteTrack

Edit `configs/bytetrack.yaml`:

- ID sering berganti saat kendaraan ketutup kendaraan lain → naikkan `track_buffer`
  (45 → 90) dan `match_thresh`.
- Muncul banyak track palsu → naikkan `new_track_thresh`.
- Kendaraan jauh/kecil sering putus → turunkan `track_low_thresh` ke 0.05.

## Verifikasi

Pipeline sudah diuji end-to-end dengan bobot COCO + video sintetis:

```bash
python src/count.py --weights yolov8n.pt --source data/videos/test_sintetis.mp4 \
  --line "0,240,640,240" --classes bus --save
```

Hasil: 1 objek, 1 ID, 1 lintasan tercatat di CSV — deteksi, tracking, dan logika
hitung terkonfirmasi jalan. Setelah training, ganti `--weights` ke `best.pt`.
