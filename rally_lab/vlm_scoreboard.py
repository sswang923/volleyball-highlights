#!/usr/bin/env python3
"""
Read the flip scoreboard with a local vision-language model (Qwen3-VL-2B, 4-bit on the GPU).

Runs in its own environment (~/vb-tools/vlm, Python 3.12, recent torch/transformers):
    cd ~/vb-tools/vlm && uv run python /mnt/c/Users/stanl/volleyball-highlights/rally_lab/vlm_scoreboard.py \
        --video /mnt/d/volleyball/stitched_20261004_game02.mp4 --times 240 600 690
    ... --checkpoints labels/20261004_game02/scoreboard_checkpoints.json   # score against known readings

Prints one JSON line per time: {"t": 600, "blue": 13, "red": 9, "raw": "...", "s": 1.8}
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# Scorer's table in the stitched 2688x1512 frame (camera behind the near end line)
CROP = "crop=420:220:2320:600"
PROMPT = ("This is a flip scoreboard on a volleyball scorer's table. It has a BLUE card set on the left "
          "and a RED card set on the right, each showing a number from 0 to 30. "
          "Read both numbers. Answer exactly in the form: blue=<number> red=<number>. "
          "If a number is hidden or you cannot read it, write ? for it.")


def load():
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, quantization_config=q, device_map="cuda:0")
    proc = AutoProcessor.from_pretrained(MODEL)
    return model, proc


def frame(video, t, tmp):
    p = Path(tmp) / f"sb_{t}.png"
    subprocess.run(["bash", "-lc", f"ffmpeg-nvenc -v error -y -hwaccel cuda -ss {t} -i '{video}' -frames:v 1 -vf {CROP} '{p}'"],
                   check=True)
    img = Image.open(p).convert("RGB")
    return img.resize((img.width * 2, img.height * 2), Image.BICUBIC)


def ask(model, proc, img):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def parse(text):
    m = re.search(r"blue\s*=\s*(\d+|\?).*?red\s*=\s*(\d+|\?)", text, re.I | re.S)
    if not m:
        return None, None
    return tuple(int(x) if x.isdigit() and int(x) <= 40 else None for x in m.groups())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--times", nargs="*", type=float, default=[])
    ap.add_argument("--checkpoints", help="scoreboard_checkpoints.json: read these times and score the answers")
    a = ap.parse_args()
    truth = {}
    times = list(a.times)
    if a.checkpoints:
        for c in json.load(open(a.checkpoints))["checkpoints"]:
            truth[c["t"]] = (c["blue"], c["red"])
            times.append(c["t"])
    model, proc = load()
    stats = {"both": 0, "one": 0, "none": 0, "wrong": 0}
    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            t0 = time.time()
            raw = ask(model, proc, frame(a.video, t, tmp))
            blue, red = parse(raw)
            row = {"t": t, "blue": blue, "red": red, "raw": raw, "s": round(time.time() - t0, 1)}
            if t in truth:
                tb, tr = truth[t]
                right = [(blue == tb) if blue is not None else None, (red == tr) if red is not None else None]
                if right == [True, True]:
                    k = "both"
                elif False in right:
                    k = "wrong"
                elif True in right:
                    k = "one"
                else:
                    k = "none"
                stats[k] += 1
                row["truth"], row["result"] = [tb, tr], k
            print(json.dumps(row), flush=True)
    if truth:
        print(json.dumps({"summary": stats, "of": len(truth),
                          "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}))


if __name__ == "__main__":
    main()
