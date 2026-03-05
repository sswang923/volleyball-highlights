# Windows Setup Guide

## Requirements

- Windows 10/11 (64-bit)
- Python 3.10 or 3.11 (recommended) — [python.org](https://www.python.org/downloads/)
- ffmpeg — [ffmpeg.org](https://ffmpeg.org/download.html#build-windows)
- (Optional but recommended) NVIDIA GPU with CUDA for much faster scanning

---

## 1. Install Python

Download and install Python 3.11 from [python.org](https://www.python.org/downloads/).

**Important:** Check **"Add Python to PATH"** during installation.

Verify:
```
python --version
```

---

## 2. Install ffmpeg

1. Download a Windows build from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/) — grab the **ffmpeg-release-essentials.zip**
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH:
   - Search "Environment Variables" in Start Menu
   - Edit `Path` under System Variables → New → `C:\ffmpeg\bin`

Verify:
```
ffmpeg -version
```

---

## 3. Install Python dependencies

### Option A — GPU (NVIDIA, recommended)

First check your CUDA version:
```
nvidia-smi
```

Install PyTorch with CUDA (adjust `cu121` to match your CUDA version — e.g. `cu118` for CUDA 11.8):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest:
```
pip install opencv-python ultralytics easyocr numpy
```

### Option B — CPU only

```
pip install -r requirements.txt
```

---

## 4. Run the script

Basic usage — scan footage for player #21:
```
python volleyball_highlights.py -i "D:\volleyball videos" -p 21 --stitch
```

With GPU acceleration (much faster OCR):
```
python volleyball_highlights.py -i "D:\volleyball videos" -p 21 --stitch --gpu
```

With parallel workers + GPU:
```
python volleyball_highlights.py -i "D:\volleyball videos" -p 21 --stitch --gpu --workers 4 --scan-interval 5.0 -o "D:\volleyball videos\highlights"
```

---

## 5. Fast crop + encode with GPU (NVENC)

If you want to crop the video to the court and re-encode using your GPU instead of CPU:

```
ffmpeg -i input.mp4 -vf "crop=2688:1012:0:500" -c:v h264_nvenc -cq 20 -preset p4 -c:a copy output_cropped.mp4
```

This finishes in **minutes** vs hours on CPU.

For AMD GPU use `-c:v h264_amf`, for Intel use `-c:v h264_qsv`.

---

## Tips

- First run will auto-download YOLOv8 weights (~6MB) — requires internet
- EasyOCR also downloads language models on first run (~100MB)
- Use `--scan-interval 5.0` for speed, `--scan-interval 1.0` for accuracy
- Use `--confidence 0.3` if detections seem too low
