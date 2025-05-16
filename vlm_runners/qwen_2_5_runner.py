import os
import gc
import json
import time
import numpy as np
import torch

from decord import cpu, VideoReader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run(cfg):
    json_file_path = cfg["json_file_path"]
    mp4_dir        = cfg["mp4_dir"]
    output_path    = cfg["output_path"]
    api_key        = cfg["mistral_key"]

## WIP
