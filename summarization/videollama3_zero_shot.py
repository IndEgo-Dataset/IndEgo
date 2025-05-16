#!/usr/bin/env python3
"""
Zero‑shot video summarisation with VideoLLaMA‑3 (no grading).

• Splits the video into N consecutive segments (default 60 s).
• Sends each segment to VideoLLaMA‑3 → summary.
• Saves one representative frame per segment.
• Writes a JSON list:
    [
      {
        "segment_index": 0,
        "start_sec": 0,
        "summary": "...",
        "ground_truth": "...",   # from annotation JSON
        "frame_path": "frames/seg000.jpg"
      },
      ...
    ]
"""

import os, json, argparse, numpy as np, torch, cv2, gc
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor


# ---------------- helpers ---------------- #
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def extract_segment_frames(vr, start_sec, seg_len, sample_fps=1, max_frames=128):
    fps = float(vr.get_avg_fps()) or 30
    idxs = np.arange(int(start_sec*fps),
                     int((start_sec+seg_len)*fps),
                     int(fps/sample_fps))
    if len(idxs) > max_frames:
        idxs = np.linspace(idxs[0], idxs[-1], max_frames, dtype=int)
    frames = [vr[i].asnumpy() for i in idxs if i < len(vr)]
    return frames, sample_fps


# ---------------- script ---------------- #
def main(a):
    # model
    device = "cuda:0"
    ckpt   = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model  = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, attn_implementation="flash_attention_2",
        device_map={"": device}, torch_dtype=torch.bfloat16)
    proc   = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

    # data
    with open(a.annotation_json) as f:
        ann = json.load(f)
    gts       = ann["annotations"]
    seg_len   = ann.get("segment_length", a.seconds)
    assert seg_len == a.seconds, "segment_length mismatch."

    vr        = VideoReader(a.mp4, ctx=cpu(0), num_threads=1)
    total_sec = len(vr) / (float(vr.get_avg_fps()) or 30)
    n_seg     = int(np.ceil(total_sec / seg_len))
    ensure_dir(a.save_dir)

    results = []
    for i in range(n_seg):
        start = i * seg_len
        frames, fps_used = extract_segment_frames(
            vr, start, seg_len, a.sample_fps, a.max_frames)
        if not frames: break

        # save thumbnail
        thumb_path = os.path.join(a.save_dir, f"seg{i:03d}.jpg")
        Image.fromarray(frames[0][:,:,::-1]).save(thumb_path)

        # VideoLLaMA‑3 prompt
        conv = [
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":[
                {"type":"video","video":{"frames":frames,"fps":fps_used}},
                {"type":"text","text":"Describe the main actions using concise noun‑verb phrases."}
            ]}
        ]
        inputs = proc(conversation=conv, add_system_prompt=True,
                      add_generation_prompt=True, return_tensors="pt")
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v
                  for k,v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        out_ids = model.generate(**inputs, max_new_tokens=a.max_tokens)
        summary = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        del inputs, out_ids; torch.cuda.empty_cache(); gc.collect()

        results.append({
            "segment_index": i,
            "start_sec":     start,
            "summary":       summary,
            "ground_truth":  gts[i] if i < len(gts) else "",
            "frame_path":    thumb_path
        })
        print(f"[seg {i}] {summary}")

        with open(a.out_json,"w") as f: json.dump(results, f, indent=2)

    print("✔ Summaries saved to", a.out_json)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mp4",            required=True)
    p.add_argument("--annotation_json",required=True)
    p.add_argument("--out_json",       required=True)
    p.add_argument("--save_dir",       required=True)
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--sample_fps", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=128)
    p.add_argument("--max_tokens", type=int, default=128)
    main(p.parse_args())
