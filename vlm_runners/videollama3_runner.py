"""
Runner for DAMO-NLP-SG/VideoLLaMA3-7B.

Requires:
  pip install flash-attn decord mistralai transformers>=4.39
"""
import os, json, time, torch, numpy as np
from mistralai import Mistral
from transformers import AutoModelForCausalLM, AutoProcessor
from utils.video_utils import sample_frames
from utils.prompt_utils import build_mcq_prompt


def run(cfg):
    jf, mp4_dir, outp, key = (cfg[k] for k in
                              ["json_file_path","mp4_dir","output_path","mistral_key"])
    client = Mistral(api_key=key)

    with open(jf, "r") as f: qa = json.load(f)

    device   = "cuda:0"
    ckpt     = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model    = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, device_map={"":device},
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    proc     = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

    results = []
    for item in qa:
        task = item["Task"]; vid_id = item["Video ID"]
        question, correct, opts = item["Question"], item["Correct Answer(s)"], item["Options"]

        # locate file
        folder = next((os.path.join(mp4_dir,x) for x in [str(task),f"{int(task):02d}"]
                       if os.path.isdir(os.path.join(mp4_dir,x))), None)
        if not folder: continue
        mp4 = next((f for f in os.listdir(folder) if vid_id in f and f.endswith(".mp4")), None)
        if not mp4: continue
        path = os.path.join(folder, mp4)

        # frames as numpy HxWxC uint8
        frames, fps = sample_frames(path, max_frames=32)
        # VideoLLaMA3 processor expects ndarray list
        conversation = [
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":[
                {"type":"video","video":{"frames":frames,"fps":fps}},
                {"type":"text", "text":build_mcq_prompt(question, opts)}]}]

        inputs = proc(conversation=conversation, add_system_prompt=True,
                      add_generation_prompt=True, return_tensors="pt")
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v
                  for k,v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        out_ids = model.generate(**inputs, max_new_tokens=128)
        resp = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        # grade with Mistral
        judge = (
            f"Q: '{question}' Options:\n" +
            "\n".join([f"{k}: {v}" for k,v in opts.items()]) +
            f"\nCorrect: {correct}. VLM: '{resp}'. Correct? Yes/No.")
        try:
            j = client.chat.complete(model='mistral-large-latest',
                                     messages=[{"role":"user","content":judge}])
            ok = 'yes' in j.choices[0].message.content.lower()
        except Exception: ok=False

        results.append({"task":task,"video":mp4,"vlm_response":resp,
                        "correct_answer":correct,"is_correct":ok,"score":int(ok)})
        with open(outp,"w") as f: json.dump(results,f,indent=2)

    print("[DONE] VideoLLaMA3 results saved to", outp)
