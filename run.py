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

from sklearn.neighbors import NearestNeighbors



#Training inputs for RGBcolors # normalized
colors = [[0., 0., 0.],
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
      [.66, .66, .66]]

#gray colors data
gray_colors = [[0.], [0.], [1.], [0.125], [0.529], [1.0], [0.33], [0.4], [0.67], [.33], [.5]]
#     torch.rand((10))

gray_colors

# +
# Network configuration
length = 4
width = 3
number_iterations = 100

input_data = convert_data(gray_colors)


# if len(input_data[0]) == 1:
#     dim = 1
# else:
dim = len(input_data[0])


# -
def convert_data(data):
    tensor_data = []
    for row in data:
        tensor_data.append(torch.FloatTensor(row))
    
    return tensor_data


input_data

torch.FloatTensor([0., 0., 1.])


map1 = MapClass(length, width, dim)

map1.map

wyliczone = (map1.map - gray_colors[0]).pow(2)
wyliczone

torch.topk(wyliczone, 1, dim=0, largest=False)





map1.get_location(11)

12/3

12%3

length = 4
width = 3

x


def get_location(number):
    row = "dupa"
    column = "dupa2"

    # if x%width == 0:
    row = int((x/width))
    column = x - (row * width)



    print(row, column)

# +
x = 4
row = "dupa"
column = "dupa2"

# if x%width == 0:
row = int((x/width))
column = x - (row * width)

    
    
print(row, column)
# -


