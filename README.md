# SOP Roll checking in Roll Wrapping

## Project Description.

Merupakan project yang bertujuan untuk melakukan pengecekan SOP secara otomatis dengan menggunakan metode **Deep Learning** (khususnya *vision* dan *recognition*).

Tujuan utamanya adalah melakukan **SOP compliance checking** dari **kamera CCTV real-time**:
- Deteksi objek penting (contoh: **person**, **helmet/PPE**, dan objek konteks lain bila dibutuhkan).
- Menilai apakah **aksi/SOP step** tertentu sudah dilakukan (fokus pada *action*, bukan identitas orang).

### Cara kerja (high-level)

1. Ambil frame dari CCTV (real-time stream).
2. Lakukan preprocessing (resize/letterbox, normalisasi) untuk input model.
3. Jalankan inference model (YOLO / model lain) untuk deteksi objek.
4. Postprocess (decode output, confidence filter, NMS, mapping koordinat kembali ke gambar original).
5. (Tahap berikutnya) Gunakan hasil deteksi (dan/atau model temporal) untuk memutuskan status SOP: done / not done / unknown. (akan dilanjutkan ketika frame sudah selesa)

### Status saat ini (kode di repo)

Saat ini repo sudah menyediakan fonda
si untuk **object detection YOLO-style** (preprocess → inference backend → postprocess) dan contoh demo inference pada gambar statis. Logic action/SOP akan dibangun di atas fondasi ini.

## Repository Layout (Current)

- `yolo_kit/` — utilitas runtime deteksi (letterbox, decode+NMS, dan backend inference: ONNX Runtime / TensorRT / TorchScript).
- `Models/` — model artifacts (contoh `.onnx`) dan `metadata.yaml` (mapping class id → name).
- `Media/` — sample media untuk demo.
- `testing_basic_detect.py` — demo sederhana untuk menjalankan deteksi pada sebuah image.
- `Action_Detection_SOP/` — placeholder package untuk logic SOP/action (akan diperluas).


## Development Environment

Project ini dikembangkan dengan menggunakan device pribadi dengan spesifikasi sebagai berikut:

- CPU   : Ryzen 5 4600H
- GPU   : Nvidia GTX 1650Ti mobile
- RAM   : 16GB DDR4

Software stack:
PYthon  : 3.10
OS      : Windows 10

Dan Pengaplikasian project ini akan dilakukan mini komputer jetson orin NX 16GB, dengan spesifikasi sebagai berikut

- GPU & CPU : Tegra Orin NX (SoC)
- RAM       : 16 GB

Software stack :
Python      :3.10
Jetpack     : 6.0 + 

## How to run the application

### Prerequisites (ringkas)

1. Python 3.10+
2. OpenCV + NumPy
3. Backend inference (pilih sesuai kebutuhan):
   - **ONNX Runtime** (laptop/dev) → `onnxruntime` atau `onnxruntime-gpu`
   - **TensorRT** (Jetson deployment) → TensorRT Python bindings (+ biasanya membutuhkan PyTorch untuk buffer di backend ini)
   - **TorchScript** (opsional) → PyTorch

### Quick demo (image detection)

Repo ini menyediakan demo untuk menjalankan deteksi pada image:

1. Pastikan dependencies sudah terinstall (repo menggunakan `uv`; jalankan sendiri sesuai environment kamu).
2. Jalankan:
   - `python3 -m Scripts.testing_basic_detect`
   - `python3 -m Scripts.visualize_detections --image Media/pedestrian.png --show --out output.jpg`

### Video / CCTV (file video atau webcam)

Untuk input video:

- `python3 -m Scripts.visualize_detections --video path/to/video.mp4 --show --out output.mp4`

Untuk webcam:

- `python3 -m Scripts.visualize_detections --webcam 0 --show`

Tips:

- Kalau FPS terlalu berat, coba proses tiap N frame: `--every 2` atau `--every 3`
- Jika FPS dari RTSP/NVR tidak terbaca (atau video output terlihat lambat), pakai:
  - `--source-fps <angka>` untuk override FPS input
  - `--video-fps-out <angka>` untuk paksa FPS video hasil simpan
- Untuk buang sesi yang terlalu pendek (noise), gunakan `--min-session-s <angka>` (0 = tidak dibuang).

## Running tests (uv)

- `uv run pytest .`
- Or: `uv run python -m pytest .\\tests\\`

## MVP-A SOP runner (operator session ROI + helmet)

Untuk mulai prototyping SOP tanpa alert (filesystem-first output):

1) Kalibrasi ROI polygon (sekali per kamera/source):
   - `python3 -m Scripts.calibrate_roi --video path/to/video.mp4 --out configs/roi.json`
   - atau RTSP: `python3 -m Scripts.calibrate_roi --rtsp "rtsp://user:pass@host/..." --out configs/roi.json`

2) (Optional) Atur timing SOP (admin):
   - `cp configs/sop_profile.example.json configs/sop_profile.json`
   - Edit `configs/sop_profile.json` (session start/end & ROI dwell seconds).

3) Jalankan SOP MVP (akan membuat `data/sessions/` dan `data/reports/`):
   - `python3 -m Scripts.run_sop_mvp --video path/to/video.mp4 --roi configs/roi.json --sop-profile configs/sop_profile.json --model Models/your_model.onnx --metadata Models/metadata.yaml --save-video`

Catatan penting:

- Model + `metadata.yaml` harus punya class **helmet** (repo saat ini memakai COCO metadata contoh, jadi kamu perlu metadata/model yang sesuai untuk PPE).
