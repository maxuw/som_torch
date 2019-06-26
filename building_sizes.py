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

data = building_sizes
batch_size = 4

length = 4
width = 3
number_iterations = 100

move_closer_coef = 0.5
iterations = 100


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
    if dim == 1:
        return map_.view(length, width)
    else:
        return map_.view(dim, length, width)


# +
def cycle(map_, training_data, display_step=False):
    for batch in training_data:
#         type(batch)
#         print(batch)
        for row in batch[0]:
#             type(row)
#             print(row)
            i_bmu = map_.find_bmu(row).item()
            map_.move_closer(i_bmu, row)
        
#     print(map_view(map_.map))
    if display_step == True:
        basic_visualization(map_display(map_.map))
        print(map_display(map_.map))
        
        

        
# -

def large_cycle(map_, training_data):
    basic_visualization(map_display(map_.map))
    print(map_display(map_.map))
    for i in range(number_iterations):
        cycle(map_, training_data)
    basic_visualization(map_display(map_.map))
    print(map_display(map_.map))


cycle(map1, training)

cycle(map1, training)

training = load_data(data)


def cycle(map_, training_data, display_step=False):
    for batch in training_data:
        t_batch = torch.stack([x for x in batch]).float().t()
        for row in t_batch:
            print(row)
            i_bmu = map_.find_bmu(row).item()
            map_.move_closer(i_bmu, row)
            
    if display_step == True:
        basic_visualization(map_display(map_.map))
        print(map_display(map_.map))


# +
for train in training:
    t_batch = torch.stack([x for x in train]).float().t()
    print(t_batch, "\n\n")
    
#     for tr in train:
#         print(tr, "\n")
# -

def cycle(map_, training_data, display_step=False):

    for batch in training_data:

        t_batch = torch.stack([x for x in batch]).float().t()

        for row in t_batch:

            print(row)

            i_bmu = map_.find_bmu(row).item()

            map_.move_closer(i_bmu, row)


map1 = MapClass(length, width, dim, move_closer_coef)

map1.map

map1.cycle(training)

map1.cycle(map1.map, training)

map1.cycle(map1.map, training)

map1.distance_matrix

map1.impact_matrix

basic_visualization(map1.map)


