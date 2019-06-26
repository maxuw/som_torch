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
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt

from map_class import MapClass



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

length = 2
width = 2
number_iterations = 100

move_closer_coef = 0.5
iterations = 100
# -


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
    if dim == 1:
        return map_.view(length, width)
    else:
        return map_.view(dim, length, width)


def large_cycle(map_, training_data):
    basic_visualization(map_display(map_.map))
    print(map_display(map_.map))
    for i in range(number_iterations):
        cycle(map_, training_data)
    basic_visualization(map_display(map_.map))
    print(map_display(map_.map))



training = load_data(data)

map1 = MapClass(length, width, dim, move_closer_coef)

map1.weights_to_map()

map1.step(training, verbose=True)

map1.step()

map1.weights



train = []
for train_ in training:
    print(train_)
    train = train_

row_data = train[0][2]
row_data

map1.weights

difference = row_data - map1.weights
difference

bmu_index = 0

map1.impact_matrix[bmu_index].view(4,1)

change = map1.impact_matrix[bmu_index].view(4,1) * difference
change

map1.weights = map1.weights + change

map1.weights

map1.impact_matrix[bmu_index]

map1.distance_matrix

torch.mm(map1.impact_matrix[bmu_index].view(4,1), change)

map1.impact_matrix

basic_visualization(map_display(map1.weights))

map_display(map1.weights)





map1.weights

difference = row_data - map1.weights

bmu_index = 0



change = map1.impact_matrix[bmu_index].view(4,1) * difference

map1.weights = map1.weights + change



map1.weights

difference = row_data - map1.weights
change = map1.impact_matrix[bmu_index].view(4,1) * difference
map1.weights = map1.weights + change
