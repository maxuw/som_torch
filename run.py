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



#buildings data
building_sizes = [[0.1, 0.3], [0.1, 0.2], [1., 1.], [0.125, 0.2], [0.529, 0.12], [1.0, 0.3], [0.33, 0.3], 
                  [0.4, 0.4], [0.67, 0.3], [.33, 0.7], [.5, 0.1]]
#     torch.rand((10))

#gray colors data
gray_colors = [[0.1], [0.], [1.], [0.125], [0.529], [1.0], [0.33], [0.4], [0.67], [.33], [.5]]
#     torch.rand((10))

def convert_data(data):
    tensor_data = []
    for row in data:
        tensor_data.append(torch.FloatTensor(row))
    
    return tensor_data


def map_view(map_):
    return map_.view(dim, length, width)


# +
# Network configuration
length = 4
width = 3
number_iterations = 100

input_data = convert_data(building_sizes)


move_closer_coef = 0.5

# if len(input_data[0]) == 1:
#     dim = 1
# else:
dim = len(input_data[0])


# -
def cycle(map_, input_data):
    for row in input_data:
        i_bmu = map1.find_bmu(row).item()
        map_.move_closer(i_bmu, row)
        
    print(map_view(map_.map))
        







map1 = MapClass(length, width, dim, move_closer_coef)

map_view(map1.map)



cycle(map1, input_data)



input_data[0]

map1.find_bmu(input_data[0]).item()

map1.move_closer(3, input_data[0])

map_view(map1.map)

input_data[1]

map1.find_bmu(input_data[1])

map1.move_closer(2, input_data[1])

map_view(map1.map)









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


