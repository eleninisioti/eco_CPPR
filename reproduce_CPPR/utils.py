import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output
import numpy as np
import random
from moviepy.editor import *
import datetime
import yaml

"""
import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output
import numpy as np
"""


def create_jzscript(project_dir):
    command = "python reproduce_CPPR/train.py " + project_dir
    with open( project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    now = datetime.datetime.now()
    scripts_dir = "jz_scripts/" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)

    script_file = scripts_dir + "/"
    for key, value in config.items():
        script_file += key + "_" + str(value)

    script_path = script_file + ".sh"

    with open(script_path, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -A imi@v100\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --cpus-per-task=8 \n")
        fh.writelines("#SBATCH --gres=gpu:1\n")
        fh.writelines("#SBATCH --hint=nomultithread\n")
        fh.writelines("#SBATCH --qos=qos_gpu-dev\n")
        fh.writelines("#SBATCH -J " + script_file + "\n")
        fh.writelines("#SBATCH -t 01:59:00\n")
        scratch_dir = "/gpfsscratch/rech/imi/utw61ti/CPPR_log/jz_logs"
        fh.writelines("#SBATCH --output=" + scratch_dir + "/%j.out\n")
        fh.writelines("#SBATCH --error=" + scratch_dir + "/%j.err\n")
        fh.writelines("module load tensorflow-gpu/py3/2.9.1\n")
        fh.writelines(command)


def merge_videos(directory, num_gens):
    import os
    gens = range(0, num_gens, 100)
    L = []

    for gen in gens:
        file_path = "projects/" + directory + "/train_" + str(gen) + ".mp4"
        video = VideoFileClip(file_path)
        L.append(video)

    final_clip = concatenate_videoclips(L)
    final_clip.to_videofile("projects/" + directory + "/total_training.mp4", fps=24, remove_temp=False)


def gini_coefficient(rewards):
    coeff = 0
    for el in rewards:
        for el2 in rewards:
            coeff += np.abs(el - el2)
    coeff = 1 - coeff / np.sum(rewards)
    return coeff


class VideoWriter:
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))


if __name__ == "__main__":
    # merge_videos("22_1_2023/multi_agent_dynamic_200_noreset_climateconstant_noreset_True", 9400)
    # merge_videos("22_1_2023/multi_agent_dynamic_200_noreset_climateno-niches_noreset_True", 9400)
    merge_videos("22_1_2023/multi_agent_dynamic_200_noreset_climateperiodic_noreset_True", 9400)
