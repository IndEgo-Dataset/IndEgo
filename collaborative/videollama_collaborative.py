#!/usr/bin/env python3
"""
Minute-wise task-understanding for TWO egocentric videos (VideoLLaMA-3).

Outputs one JSON file of the form
[
  { "segment_index": 0,
    "start_sec": 0,
    "video_a": {
        "self_action": "...",
        "coworker_action": "...",
        "self_role": "teacher/student/collaborator",
        "coworker_role": "..."
    },
    "video_b": {...}
  },
  ...
]
"""

import os, json, argparse, numpy as np, torch, gc
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

# ---------- helpers ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def extract(vr, start, dur, fps_sample=1, max_frames=128):
    fps = float(vr.get_avg_fps()) or 30
    idxs = np.arange(int(start*fps), int((start+dur)*fps), int(fps/fps_sample))
    if len(idxs) > max_frames:
        idxs = np.linspace(idxs[0], idxs[-1], max_frames, dtype=int)
    return [vr[i].asnumpy() for i in idxs if i < len(vr)], fps_sample

def prompt_template():
    return ("For each minute clip you will see *your* egocentric view and that of "
            "your coworker (recorded separately). "
            "Return a JSON with keys:\n"
            "  self_action: <noun-verb phrase>\n"
            "  coworker_action: <noun-verb phrase>\n"
            "  self_role: teacher / student / collaborator\n"
            "  coworker_role: teacher / student / collaborator\n"
            "Respond ONLY in JSON.")

# ---------- script ----------
def main(a):
    ckpt  = "DAMO-NLP-SG/VideoLLaMA3-7B"
    device= "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, attn_implementation="flash_attention_2",
        device_map={"":device}, torch_dtype=torch.bfloat16)
    proc  = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

    vr_a, vr_b = VideoReader(a.video_a, ctx=cpu(0)), VideoReader(a.video_b, ctx=cpu(0))
    dur = a.minutes * 60
    total = min(len(vr_a), len(vr_b)) / (float(vr_a.get_avg_fps()) or 30)
    segments = int(np.ceil(total / dur))

    ensure_dir(a.save_dir)
    results=[]
    for i in range(segments):
        st = i*dur
        frames_a, fps_a = extract(vr_a, st, dur, a.sample_fps, a.max_frames)
        frames_b, fps_b = extract(vr_b, st, dur, a.sample_fps, a.max_frames)
        if not frames_a or not frames_b: break

        # save thumbs (optional debugging)
        Image.fromarray(frames_a[0][:,:,::-1]).save(f"{a.save_dir}/A_seg{i:03d}.jpg")
        Image.fromarray(frames_b[0][:,:,::-1]).save(f"{a.save_dir}/B_seg{i:03d}.jpg")

        conv = [
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":[
                {"type":"video","video":{"frames":frames_a,"fps":fps_a}},
                {"type":"video","video":{"frames":frames_b,"fps":fps_b}},
                {"type":"text","text":prompt_template()}
            ]}
        ]
        inputs = proc(conversation=conv, add_system_prompt=True,
                      add_generation_prompt=True, return_tensors="pt")
        inputs = {k:(v.to(device) if isinstance(v,torch.Tensor) else v)
                  for k,v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"]=inputs["pixel_values"].to(torch.bfloat16)
        out = model.generate(**inputs, max_new_tokens=a.max_tokens)
        txt = proc.batch_decode(out, skip_special_tokens=True)[0].strip()

        # attempt to parse JSON; if fails, keep raw text
        try:  parsed=json.loads(txt)
        except Exception: parsed={"raw_response":txt}

        results.append({"segment_index":i,"start_sec":st,
                        "video_a":parsed.get("video_a",{}),
                        "video_b":parsed.get("video_b",{})})

        del inputs, out; torch.cuda.empty_cache(); gc.collect()
        with open(a.out_json,"w") as f: json.dump(results,f,indent=2)
        print(f"[seg {i}] processed")

    print("✔ saved to", a.out_json)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--video_a", required=True, help="Person A mp4")
    p.add_argument("--video_b", required=True, help="Person B mp4")
    p.add_argument("--annotation_json", required=True,
                   help="Ground-truth JSON for evaluation phase (not used here)")
    p.add_argument("--out_json", required=True, help="save predictions here")
    p.add_argument("--save_dir", required=True, help="where to save thumbs")
    p.add_argument("--minutes", type=int, default=1, help="segment length, minutes")
    p.add_argument("--sample_fps", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=128)
    p.add_argument("--max_tokens", type=int, default=256)
    main(p.parse_args())
