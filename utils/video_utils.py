import numpy as np
from decord import VideoReader, cpu

def sample_frames(video_path, max_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idxs  = np.linspace(0, total-1, num=min(max_frames,total), dtype=np.int32)
    return np.stack([vr[i].asnumpy() for i in idxs]), float(vr.get_avg_fps())
