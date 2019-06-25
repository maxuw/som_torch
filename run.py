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

from map_class import MapClass



#gray colors data
gray_colors = [[0.1], [0.], [1.], [0.125], [0.529], [1.0], [0.33], [0.4], [0.67], [.33], [.5]]
#     torch.rand((10))

gray_colors


def convert_data(data):
    tensor_data = []
    for row in data:
        tensor_data.append(torch.FloatTensor(row))
    
    return tensor_data


# +
# Network configuration
length = 4
width = 3
number_iterations = 100

input_data = convert_data(gray_colors)


move_closer_coef = 0.5

# if len(input_data[0]) == 1:
#     dim = 1
# else:
dim = len(input_data[0])
# -
input_data[0].shape

torch.FloatTensor([0., 0., 1.])


map1 = MapClass(length, width, dim, move_closer_coef)

map1.map

map1.map

map1.find_bmu(input_data[0])

map1.map[5]

input_data[0]

change = map1.map[5] - input_data[0]

map1.map[5] = map1.map[5] - (change * move_closer_coef)

map1.map

map1.move_closer(0, input_data[0])

map1


