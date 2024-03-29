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

import numpy as np
import matplotlib.pyplot as plt

from map_class import MapClass

import torch.utils.data

#buildings data
building_sizes = [[0.1, 0.3], [0.1, 0.2], [1., 1.], [0.125, 0.2], [0.529, 0.12], [1.0, 0.3], [0.33, 0.3], 
                  [0.4, 0.4], [0.67, 0.3], [.33, 0.7], [.5, 0.1]]
#     torch.rand((10))

#gray colors data
gray_colors = [[0.1], [0.], [1.], [0.125], [0.529], [1.0], [0.33], [0.4], [0.67], [.33], [.5]]
#     torch.rand((10))

# +
# Network configuration

data = gray_colors
batch_size = 4

length = 4
width = 3
number_iterations = 100

move_closer_coef = 0.5

# if len(input_data[0]) == 1:
#     dim = 1
# else:

dim = 0


# -
def basic_visualization(map_):
    plt.imshow(map_);
    plt.colorbar()
    plt.show()


# +
trainloader = ""

def load_data(data, batch_size=4, shuffle=False):
    global dim
    dim = len(data[0])
    
    trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    return trainloader


# -

def map_view_for_coding(map_):
    return torch.transpose(map_, 0, 1).view(dim, length, width)
#     return map_.view(dim, length, width)


def map_display(map_):
#     return torch.transpose(map_, 0, 1).view(dim, length, width)
    return map_.view(dim, length, width)


def cycle(map_, training_data):
    for batch in training_data:
        for row in batch:
            i_bmu = map1.find_bmu(row).item()
            map_.move_closer(i_bmu, row)
        
    print(map_view(map_.map))   


training = load_data(data)

map1 = MapClass(length, width, dim, move_closer_coef)

map1.map



map2.map.view(dim, length, width)

map1 = MapClass(length, width, dim, move_closer_coef)

map1.map

basic_visualization(numpy_array)


