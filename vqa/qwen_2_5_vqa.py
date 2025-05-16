import os
import gc
import json
import time
import numpy as np
import torch

from decord import cpu, VideoReader
from mistralai import Mistral
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# -------------------------------
# CONFIG / PATHS
# -------------------------------
json_file_path = ""
mp4_dir = ""
output_path = ""

api_key = ''  # Mistral key
client = Mistral(api_key=api_key)

# -------------------------------
# HYPERPARAMS
# -------------------------------
MAX_FRAMES = 16               # Hard cap on total frames from each video
MAX_NEW_TOKENS = 64           # Generate fewer tokens to save memory
MAX_PIXELS = 480 * 480        # Additional cap on resolution to save memory

# -------------------------------
# LOAD DATA
# -------------------------------
with open(json_file_path, 'r') as f:
    questions_data = json.load(f)

# -------------------------------
# LOAD MODEL & PROCESSOR
# -------------------------------
# 1. Use flash_attention_2 + half precision to reduce VRAM usage
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto", device_map="auto"
)
model.eval()

# 2. Create processor with optional min/max pixel settings
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    max_pixels=MAX_PIXELS,
)

# -------------------------------
# FUNCTION: PROCESS VIDEO
# -------------------------------
def get_processed_video_frames(video_path, max_frames=16):
    """
    Returns up to `max_frames` frames evenly spaced throughout the video.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    if total_frames == 0:
        return np.empty((0,)), 0.0

    # We'll sample exactly `max_frames` indices, spaced evenly.
    # For a short video with fewer than max_frames, we get them all.
    indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int32)
    indices = np.unique(indices)  # ensure unique integer frames
    frames = [vr[idx].asnumpy() for idx in indices]
    frames = np.stack(frames)
    fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 1.0
    return frames, fps

# -------------------------------
# MAIN LOOP
# -------------------------------
results = []

for question_entry in questions_data:
    task_number = question_entry["Task"]
    video_id = question_entry["Video ID"]
    question_text = question_entry["Question"]
    correct_answer = question_entry["Correct Answer(s)"]
    options = question_entry["Options"]

    # Locate the task folder
    folder_names = [str(task_number), f"{int(task_number):02d}"]
    task_folder = None
    for folder_name in folder_names:
        potential_folder = os.path.join(mp4_dir, folder_name)
        if os.path.isdir(potential_folder):
            task_folder = potential_folder
            break

    if not task_folder:
        print(f"Task folder for {task_number} not found. Skipping task {task_number}.")
        continue

    matching_videos = [
        file for file in os.listdir(task_folder)
        if video_id in file and file.endswith(".mp4")
    ]
    if not matching_videos:
        print(f"No video file matching {video_id} found in folder {task_folder}. Skipping.")
        continue

    # We assume the first matching video is the desired one
    video_path = os.path.join(task_folder, matching_videos[0])
    print(f"Processing video: {video_path}")

    # 1) READ & SAMPLE VIDEO FRAMES
    # Limit frames to save memory
    processed_video, fps = get_processed_video_frames(video_path, max_frames=MAX_FRAMES)

    # 2) BUILD PROMPT
    options_text = (
        f"a) {options['A']}\n"
        f"b) {options['B']}\n"
        f"c) {options['C']}\n"
        f"d) {options['D']}\n"
        f"e) {options['E']}\n"
    )
    question_prompt = (
        f"{question_text}\n"
        "You are watching an egocentric recording captured by a user performing a task. "
        "Analyze the input video, the objects, and the action, and think carefully about the question. "
        "Select one of the options (or two if both are correct) and explain your choice in short.\n"
        f"{options_text}"
    )

    # 3) CREATE MESSAGES
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": MAX_PIXELS,
                    "fps": fps,
                },
                {"type": "text", "text": question_prompt},
            ],
        }
    ]

    # 4) PREPARE MODEL INPUT
    #    Must place the model call inside an inference_mode context to save memory
    with torch.inference_mode():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move to GPU
        inputs = inputs.to("cuda", non_blocking=True)

        # 5) GENERATE
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        # TRIM
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        vlm_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    # 6) CLEAN UP GPU TENSORS RIGHT AWAY
    del inputs
    del generated_ids
    del generated_ids_trimmed
    del image_inputs
    del video_inputs
    torch.cuda.empty_cache()
    gc.collect()

    # 7) OUTPUT
    print("\n=========================")
    print(f"Task Number: {task_number}")
    print(f"--> Question: {question_text}")
    print(f"--> Options: {options}")
    print(f"--> Correct Answer: {correct_answer}")
    print(f"--> VLM Response: {vlm_response}")

    # 8) MISTRAL CHECK
    if vlm_response:
        correct_answers = correct_answer.split(",") if "," in correct_answer else [correct_answer]
        acceptable_answers_text = ", ".join(f"'{ans.strip()}'" for ans in correct_answers)

        mistral_judgment_prompt = (
            f"The question is: '{question_text}'. The options are:\n"
            f"A: {options['A']}\n"
            f"B: {options['B']}\n"
            f"C: {options['C']}\n"
            f"D: {options['D']}\n"
            f"Acceptable correct answer(s) are: {acceptable_answers_text}. "
            f"The VLM responded with: '{vlm_response}'. "
            "Does the VLM's response contain at least one of these acceptable answers? "
            "Answer 'Yes' or 'No'."
        )

        # Retry Mistral call if needed
        while True:
            try:
                response = client.chat.complete(
                    model='mistral-large-latest',
                    messages=[{"role": "user", "content": mistral_judgment_prompt}]
                )
                judgment = response.choices[0].message.content.strip().lower()
                print(f"--> Mistral Judgment Raw Response: {judgment}")

                is_correct = "yes" in judgment
                score = 1 if is_correct else 0
                break

            except Exception as e:
                print(f"Mistral API error: {e}")
                time.sleep(10)
    else:
        judgment, is_correct, score = "No response from VLM", False, 0

    print(f"--> Judgment by Mistral: {judgment if vlm_response else 'N/A'}")
    print(f"--> Score: {score}")

    # 9) SAVE RESULTS
    results.append({
        "task_number": task_number,
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "vlm_response": vlm_response,
        "is_correct": is_correct,
        "score": score,
        "video_file": matching_videos[0]
    })

    # Write out after each item
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Also release big arrays (CPU side)
    del processed_video

    # One more round of cleanup after everything
    torch.cuda.empty_cache()
    gc.collect()

print(f"\nResults saved successfully to {output_path}")
