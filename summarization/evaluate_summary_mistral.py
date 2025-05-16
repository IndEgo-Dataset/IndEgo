#!/usr/bin/env python3
"""
Judges the summaries with Mistral‑Large and prints Accuracy & F1.

Inputs
------
--summary_json : output from summarise_videollama3.py
--mistral_key  : your API key
"""

import json, argparse, time
from mistralai import Mistral
from collections import Counter

def main(a):
    with open(a.summary_json) as f:
        data = json.load(f)

    client = Mistral(api_key=a.mistral_key)
    y_true, y_pred = [], []

    for seg in data:
        gt  = seg["ground_truth"]
        sum = seg["summary"]

        prompt = (f"Ground‑truth actions: \"{gt}\".\n"
                  f"Model summary: \"{sum}\".\n"
                  "Does the summary capture the ground‑truth actions? "
                  "Answer Yes or No (first word only).")
        while True:
            try:
                resp = client.chat.complete(
                    model='mistral-large-latest',
                    messages=[{"role":"user","content":prompt}]
                )
                ans = resp.choices[0].message.content.lower().strip()
                y_true.append(1)            # every segment expected correct
                y_pred.append(1 if ans.startswith("yes") else 0)
                break
            except Exception as e:
                print("Mistral error, retrying:", e); time.sleep(5)

    # metrics
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    acc = sum(1 for t,p in zip(y_true,y_pred) if t==p)/len(y_true)
    prec = tp/(tp+fp) if tp+fp else 0
    rec  = tp/(tp+fn) if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0

    print(f"\nSegments evaluated : {len(y_true)}")
    print(f"Accuracy           : {acc:.3f}")
    print(f"Precision          : {prec:.3f}")
    print(f"Recall             : {rec:.3f}")
    print(f"F1‑score           : {f1:.3f}")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--summary_json", required=True)
    pa.add_argument("--mistral_key",  required=True)
    main(pa.parse_args())
