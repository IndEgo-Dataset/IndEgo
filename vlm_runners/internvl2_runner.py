"""
Runner for OpenGVLab/InternVL2_5-8B.

Heavy pre/post-processing is unchanged from your original script
but wrapped in run(cfg).
"""
import os, json, time, torch, numpy as np
from mistralai import Mistral
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from utils.prompt_utils import build_mcq_prompt


# ---------- transform helpers  ----------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_video_tiles(path, num_segments=8, size=448):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(vr)-1; fps = float(vr.get_avg_fps())
    idxs  = np.linspace(0,total,num_segments,dtype=int)
    tfm   = build_transform(size)
    tiles, patches = [], []
    for i in idxs:
        img = Image.fromarray(vr[i].asnumpy())
        tile = tfm(img).unsqueeze(0)     # single tile
        tiles.append(tile)
        patches.append(1)
    return torch.cat(tiles), patches


def run(cfg):
    jf, mp4_dir, outp, key = (cfg[k] for k in
                              ["json_file_path","mp4_dir","output_path","mistral_key"])
    client = Mistral(api_key=key)

    with open(jf) as f: qa = json.load(f)

    path = 'OpenGVLab/InternVL2_5-8B'
    model = AutoModel.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tok   = AutoTokenizer.from_pretrained(path, trust_remote_code=True,
                                          use_fast=False)

    results=[]
    for item in qa:
        task, vid_id = item["Task"], item["Video ID"]
        question, correct, opts = item["Question"], item["Correct Answer(s)"], item["Options"]

        folder = next((os.path.join(mp4_dir,x) for x in [str(task),f"{int(task):02d}"]
                       if os.path.isdir(os.path.join(mp4_dir,x))), None)
        if not folder: continue
        mp4 = next((f for f in os.listdir(folder) if vid_id in f and f.endswith(".mp4")), None)
        if not mp4: continue
        vpath = os.path.join(folder, mp4)

        pixel_vals, patch_counts = load_video_tiles(vpath, num_segments=8)
        pixel_vals = pixel_vals.to(torch.bfloat16).cuda()

        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(patch_counts))])
        prompt = video_prefix + build_mcq_prompt(question, opts)
        gen_cfg = dict(max_new_tokens=1024, do_sample=True)

        vlm_resp, _ = model.chat(tok, pixel_vals, prompt, gen_cfg,
                                 num_patches_list=patch_counts, history=None,
                                 return_history=True)

        # grade with Mistral
        j_prompt = ("Q: '"+question+"' Options:\n" +
                    "\n".join([f"{k}: {v}" for k,v in opts.items()]) +
                    f"\nCorrect: {correct}. VLM: '{vlm_resp}'. Correct? Yes/No.")
        try:
            j = client.chat.complete(model='mistral-large-latest',
                                     messages=[{"role":"user","content":j_prompt}])
            ok = 'yes' in j.choices[0].message.content.lower()
        except Exception: ok=False

        results.append({"task":task,"video":mp4,"vlm_response":vlm_resp,
                        "correct_answer":correct,"is_correct":ok,"score":int(ok)})
        with open(outp,"w") as f: json.dump(results,f,indent=2)

    print("[DONE] InternVL2 results written to", outp)
