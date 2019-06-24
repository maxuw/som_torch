# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch

from map_class import Map





#Training inputs for RGBcolors # normalized
colors = torch.tensor(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])

#gray colors data
gray_colors = torch.rand((10))

gray_colors

input_data.dim()

# +
# Network configuration
length = 3
width = 3
number_iterations = 100

input_data = gray_colors

if input_data.dim() == 1:
    dim = 1
else:
    dim = len(input_data[0])
# -

map1 = Map()

map1.dupa()
