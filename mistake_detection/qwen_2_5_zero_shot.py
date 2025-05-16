#!/usr/bin/env python3
"""
Mistake-detection benchmark (Qwen-2-5-VL).

JSON format expected
--------------------
Example: https://huggingface.co/datasets/vivek9chavan/IndEgo_Demo/tree/main/Mistake_Detection/Task_10
[
  {
    "Task"        : "Trolley Loading",
    "Video ID"    : "User_16_410_2110_2_480",
    "mp4_folder"  : "1_Assembly",          # (or leave blank → script infers)
    "step_annotations": {                  # ground-truth: performed? True/False
        "open hatch"      : true,
        "put on gloves"   : true,
        "load trolley"    : true,
        "close hatch"     : true,
        "test"            : false
    },
    "mistake_annotations": {               # ground-truth: mistake occurred?
        "didn't open hatch": false
    }
  },
  ...
]

Edit the PATHS block below, or call with --json / --mp4_dir / --out.
"""
import os, gc, json, time, argparse, numpy as np, torch
from decord import cpu, VideoReader
from mistralai import Mistral
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ------------------------------- helpers ------------------------------- #
def sample_frames(video_path, max_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    n  = len(vr)
    if n == 0:
        return np.empty((0,)), 1.0
    idxs = np.linspace(0, n-1, num=min(max_frames, n), dtype=np.int32)
    frames = np.stack([vr[i].asnumpy() for i in idxs])
    return frames, float(vr.get_avg_fps()) or 1.0


def make_prompt(statement):
    return (f"{statement}\n"
            "Answer strictly 'Yes' or 'No' first, then give a one-sentence reason.")


# ------------------------------- main ------------------------------- #
def main(cfg):
    # ---------- config ---------- #
    json_path   = cfg.json
    mp4_root    = cfg.mp4_dir
    out_path    = cfg.out
    mistral_key = cfg.mistral

    MAX_FRAMES, MAX_TOK, MAX_PIX = 16, 64, 480*480

    # ---------- load data ---------- #
    with open(json_path, "r") as f:
        items = json.load(f)

    # ---------- models ---------- #
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto").eval()
    proc  = AutoProcessor.from_pretrained(model_name, max_pixels=MAX_PIX)
    judge = Mistral(api_key=mistral_key)

    results = []

    for it in items:
        task      = it["Task"]
        vid_id    = it["Video ID"]
        folder    = it.get("mp4_folder")  # optional shortcut
        steps_gt  = it["step_annotations"]
        mistakes_gt = it["mistake_annotations"]

        # locate mp4 folder automatically if not given
        if not folder:
            folder = next((os.path.join(mp4_root, x)
                           for x in os.listdir(mp4_root)
                           if os.path.isdir(os.path.join(mp4_root, x))
                           and vid_id in " ".join(os.listdir(os.path.join(mp4_root, x)))),
                          None)
        if not folder:
            print(f"[WARN] folder for video {vid_id} not found — skipping")
            continue

        mp4 = next((f for f in os.listdir(folder) if vid_id in f and f.endswith(".mp4")), None)
        if not mp4:
            print(f"[WARN] {vid_id} *.mp4 not in {folder}")
            continue
        vpath = os.path.join(folder, mp4)

        frames, fps = sample_frames(vpath, max_frames=MAX_FRAMES)

        # --------------- build (video + question) messages --------------- #
        # iterate over each binary question (step performed? / mistake occurred?)
        qa_pairs = []

        for step, done in steps_gt.items():
            q = f"Did the user **{step}**?"
            qa_pairs.append((q, "yes" if done else "no"))

        for mistake, occ in mistakes_gt.items():
            q = f"Did the user **{mistake}**?"
            qa_pairs.append((q, "yes" if occ else "no"))

        for q_text, gt in qa_pairs:
            prompt = make_prompt(q_text)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video",
                     "video": vpath,
                     "max_pixels": MAX_PIX,
                     "fps": fps},
                    {"type": "text",  "text": prompt}
                ]}]

            # ---------- Qwen inference ---------- #
            with torch.inference_mode():
                chat_txt = proc.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                img_inp, vid_inp = process_vision_info(messages)
                inputs = proc(text=[chat_txt], images=img_inp, videos=vid_inp,
                              padding=True, return_tensors="pt").to("cuda")
                out_ids = model.generate(**inputs, max_new_tokens=MAX_TOK)
                resp = proc.batch_decode(
                    [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)],
                    skip_special_tokens=True)[0].strip()

            # ------------- free VRAM ------------- #
            del inputs, out_ids, img_inp, vid_inp
            torch.cuda.empty_cache(); gc.collect()

            # ---------- ask Mistral to grade ---------- #
            grade_prompt = (
                f"Question: {q_text}\n"
                f"Ground truth answer: '{gt}'. "
                f"Model answered: '{resp}'. "
                "Does the model's first 'Yes'/'No' match the ground truth? "
                "Answer 'Yes' or 'No'.")
            try:
                g = judge.chat.complete(
                        model='mistral-large-latest',
                        messages=[{"role":"user","content":grade_prompt}]
                    )
                is_correct = 'yes' in g.choices[0].message.content.lower()
            except Exception as e:
                print("[Mistral error]", e)
                is_correct = False

            results.append({
                "task": task,
                "video": mp4,
                "question": q_text,
                "ground_truth": gt,
                "vlm_response": resp,
                "is_correct": is_correct,
                "score": int(is_correct)
            })
            # incremental save
            with open(out_path, "w") as f: json.dump(results, f, indent=2)

            print(f"[{task}] {q_text} → VLM='{resp[:40]}...'  ✓{is_correct}")

    print("\n[DONE] Mistake-detection results saved to", out_path)


# ------------------------------- CLI ------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json",      required=True, help="Path to mistake JSON file")
    p.add_argument("--mp4_dir",   required=True, help="Root folder of .mp4 directories")
    p.add_argument("--out",       required=True, help="Output JSON file")
    p.add_argument("--mistral",   required=True, help="Mistral API key")
    args = p.parse_args()
    main(args)
