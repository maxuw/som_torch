import torch
from torch.nn.modules.distance import PairwiseDistance


class MapClass:

    def __init__(self, length, width, node_dimension, move_closer_coef):
        # print("dupa")
        self.length = length
        self.width = width
        self.node_dimenstion = node_dimension
        self.move_closer_coef = move_closer_coef


        self.map = self.initialize_map(self.length, self.width, self.node_dimenstion)
        self.location = self.initialize_locations(self.map)

        # self.initialize_location(self.length, self.width, self.node_dimenstion)

    def initialize_map(self, length, width, dimention):
        map_init = torch.rand((length * width, dimention))


        return map_init

    def get_location(self, node_number):
        row = "dupa"
        column = "dupa2"

        # if x%width == 0:
        row = int((node_number / self.width))
        column = node_number - (row * self.width)

        print(row, column)
        return(row, column)

    # returns index - topk[1];
    def find_bmu(self, tensor_row_data):
        calc = (self.map - tensor_row_data).pow(2)
        # print(calc)
        summed_rows = (torch.sum(calc, dim=1))
        # print(summed_rows)
        topk = torch.topk(summed_rows, 1, dim=0, largest=False)
        # print(topk)
        return topk[1]

    def move_closer(self, bmu_index, tensor_row_data):
        change = self.map[bmu_index] - tensor_row_data

        self.map[bmu_index].add_(-(change * self.move_closer_coef))

    def initialize_locations(self, map_):
        locations = []
        for i in range(len(map_)):
            location = self.get_location(i)
            locations.append(location)
            # print(location)
        return locations

    def create_distance_matrix(self, locations, length, width):
        distance_matrix = torch.zeros(length * width, length * width)

        pair_dist = nn.PairwiseDistance(p=2)
        distances = pair_dist()


        >>> pdist = nn.PairwiseDistance(p=2)
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> output = pdist(input1, input2)

        return distance_matrix

    def calculate_distance(self):