"""
Runner for Qwen-2-5 VL-7B-Instruct.

Expects a cfg dict with keys:
    json_file_path, mp4_dir, output_path, mistral_key
"""
import os, gc, json, time, numpy as np, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from mistralai import Mistral
from decord import cpu, VideoReader
from utils.video_utils import sample_frames
from utils.prompt_utils import build_mcq_prompt


# ---------- Hyper-params ----------
MAX_FRAMES      = 16
MAX_NEW_TOKENS  = 64
MAX_PIXELS      = 480 * 480


def run(cfg):
    json_file_path = cfg["json_file_path"]
    mp4_dir        = cfg["mp4_dir"]
    output_path    = cfg["output_path"]
    api_key        = cfg["mistral_key"]

    client = Mistral(api_key=api_key)

    with open(json_file_path, "r") as f:
        qa_list = json.load(f)

    # Model & processor
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_name, max_pixels=MAX_PIXELS
    )

    results = []
    for item in qa_list:
        task_nr   = item["Task"]
        video_id  = item["Video ID"]
        question  = item["Question"]
        correct   = item["Correct Answer(s)"]
        options   = item["Options"]

        # ------ locate video ------
        folder = None
        for cand in [str(task_nr), f"{int(task_nr):02d}"]:
            p = os.path.join(mp4_dir, cand)
            if os.path.isdir(p):
                folder = p
                break
        if folder is None:
            print(f"[WARN] folder for task {task_nr} not found")
            continue

        mp4_file = next((f for f in os.listdir(folder)
                         if video_id in f and f.endswith(".mp4")), None)
        if mp4_file is None:
            print(f"[WARN] video {video_id} not found in {folder}")
            continue
        video_path = os.path.join(folder, mp4_file)

        # ------ video frames ------
        frames, fps = sample_frames(video_path, max_frames=MAX_FRAMES)

        # ------ prompt ------
        prompt = build_mcq_prompt(question, options)
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path,
                 "max_pixels": MAX_PIXELS, "fps": fps},
                {"type": "text",  "text": prompt},
            ],
        }]

        with torch.inference_mode():
            text  = processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info  # local util from Qwen
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
                               padding=True, return_tensors="pt").to("cuda")

            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            resp = processor.batch_decode(
                [oid[len(iid):] for iid, oid in zip(inputs.input_ids, out_ids)],
                skip_special_tokens=True)[0].strip()

        # memory clean-up
        del inputs, out_ids, img_inputs, vid_inputs
        torch.cuda.empty_cache(); gc.collect()

        # ------ Mistral grading ------
        judge_prompt = (
            f"The question is: '{question}'. Options:\n"
            + "\n".join([f"{k}: {v}" for k, v in options.items()]) + "\n"
            f"Correct answers: {correct}. "
            f"VLM said: '{resp}'. Does it contain a correct answer? Answer Yes/No."
        )
        try:
            j = client.chat.complete(model='mistral-large-latest',
                                     messages=[{"role": "user",
                                                "content": judge_prompt}])
            is_correct = 'yes' in j.choices[0].message.content.lower()
        except Exception as e:
            print("[Mistral error]", e)
            is_correct = False

        results.append({
            "task": task_nr, "video": mp4_file,
            "question": question, "vlm_response": resp,
            "correct_answer": correct,
            "is_correct": is_correct, "score": int(is_correct)
        })
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"[DONE] Results written to {output_path}")
