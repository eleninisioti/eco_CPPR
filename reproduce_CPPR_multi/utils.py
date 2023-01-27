
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output
import numpy as np
import random
from moviepy.editor import *
"""
import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output
import numpy as np
"""

def merge_videos(directory, num_gens):
  import os
  gens = range(0, num_gens, 100)
  L =[]

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
      coeff += np.abs(el-el2)
  coeff = 1 -coeff/np.sum(rewards)
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
      img = np.uint8(img.clip(0, 1 ) *255)
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


def compute_imitation(actions,  num_hard_coded):
  actions = np.array(actions)
  group_imitation = 0
  hard_action = actions[0]
  for action in actions[num_hard_coded:]:
    identical_moves = np.sum([1 for idx, el in enumerate(action) if el == hard_action[idx]])
    if identical_moves == len(hard_action):
      group_imitation += 1

  group_imitation = group_imitation/(len(actions)-num_hard_coded)


  return group_imitation



if __name__ == "__main__":
  #merge_videos("22_1_2023/multi_agent_dynamic_200_noreset_climateconstant_noreset_True", 9400)
  #merge_videos("22_1_2023/multi_agent_dynamic_200_noreset_climateno-niches_noreset_True", 9400)
  merge_videos("22_1_2023/multi_agent_dynamic_200_noreset_climateperiodic_noreset_True", 9400)