import torch

# class Map:
#
#     def dupa(self):
#         print("dupa")


class MapClass:

    def __init__(self, length, width, node_dimension):
        # print("dupa")
        self.length = length
        self.width = width
        self.node_dimenstion = node_dimension

        # I'm defining

        self.map = self.initialize_map(self.length, self.width, self.node_dimenstion)

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

    def find_bmu(self, tensor_row_data):
        calc = (self.map - tensor_row_data).pow(2)
        topk = torch.topk(calc, 1, dim=0, largest=False)
        return topk[1]
