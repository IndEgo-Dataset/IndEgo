import numpy as np
import os

def build_mcq_prompt(q_text, options_dict):
    opts = "\n".join([f"{k.lower()}) {v}" for k, v in options_dict.items()])
    return (f"{q_text}\n"
            "You are watching an egocentric recording captured by a user performing a task. "
            "Analyse the video carefully, then choose the correct option(s) and justify briefly.\n"
            f"{opts}")
