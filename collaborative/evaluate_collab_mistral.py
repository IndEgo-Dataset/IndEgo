#!/usr/bin/env python3
"""
Grades collaborative-understanding JSON with Mistral-Large.

Ground-truth annotation JSON (one entry per segment):
[
  { "video_a": { "self_action": "...", "coworker_action": "...",
                 "self_role": "...", "coworker_role": "..." },
    "video_b": { ... } },
  ...
]
"""

import json, argparse, time
from mistralai import Mistral

def judge_pair(gt, pred, client):
    prompt = (f"Ground truth:\n{json.dumps(gt)}\n\n"
              f"Prediction:\n{json.dumps(pred)}\n\n"
              "Do all four fields (self_action, coworker_action, self_role, "
              "coworker_role) match semantically? Answer Yes or No only.")
    while True:
        try:
            r = client.chat.complete(
                model='mistral-large-latest',
                messages=[{"role":"user","content":prompt}])
            return r.choices[0].message.content.lower().startswith("yes")
        except Exception as e:
            print("Mistral error:", e); time.sleep(5)

def main(a):
    with open(a.pred_json) as f: preds = json.load(f)
    with open(a.gt_json)   as f: gt   = json.load(f)

    client = Mistral(api_key=a.key)
    correct=0
    for seg_idx,(p,g) in enumerate(zip(preds,gt)):
        ok_a = judge_pair(g["video_a"], p["video_a"], client)
        ok_b = judge_pair(g["video_b"], p["video_b"], client)
        correct += ok_a + ok_b
        print(f"Segment {seg_idx}: A={'✓' if ok_a else '✗'}  B={'✓' if ok_b else '✗'}")

    total = 2*len(gt)
    acc   = correct/total
    print(f"\nOverall Accuracy: {acc:.3f}  ({correct}/{total})")

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--pred_json", required=True, help="output of VL script")
    pa.add_argument("--gt_json",   required=True, help="ground-truth annotations")
    pa.add_argument("--key",       required=True, help="Mistral API key")
    main(pa.parse_args())
