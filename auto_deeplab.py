import torch
import torch.nn as nn
import numpy as np
import model_search
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F
from operations import *

class AutoDeeplab (nn.Module) :

    def __init__(self, num_classes, num_layers, criterion, num_channel = 40, multiplier = 5, step = 5, crop_size=None, cell=model_search.Cell):
        super(AutoDeeplab, self).__init__()
        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._multiplier = multiplier
        self._num_channel = num_channel
        self._criterion = criterion
        self._crop_size = crop_size
        self._arch_param_names = ["alphas_cell", "alphas_network"]
        self._initialize_alphas ()
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU ()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU ()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU ()
        )

        C_prev_prev = 64
        C_prev = 128
        for i in range (self._num_layers) :
        # def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate) : rate = 0 , 1, 2  reduce rate

            if i == 0 :
                cell1 = cell (self._step, self._multiplier, -1, C_prev, self._num_channel, 1)
                cell2 = cell (self._step, self._multiplier, -1, C_prev, self._num_channel * 2, 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1 :
                cell1_1 = cell (self._step, self._multiplier, C_prev, self._num_channel, self._num_channel, 1)
                cell1_2 = cell (self._step, self._multiplier, C_prev, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell (self._step, self._multiplier, -1, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell (self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 2, 1)

                cell3 = cell (self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 4, 2)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell3]

            elif i == 2 :
                cell1_1 = cell (self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell (self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell (self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell (self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 4, 1)

                cell4 = cell (self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 8, 2)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell4]



            elif i == 3 :
                cell1_1 = cell (self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell (self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 4, self._num_channel * 4, 1)
                cell3_3 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 8, self._num_channel * 4, 0)


                cell4_1 = cell (self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 8, 2)
                cell4_2 = cell (self._step, self._multiplier, -1, self._num_channel * 8, self._num_channel * 8, 1)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]

            else :
                cell1_1 = cell (self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell (self._step, self._multiplier, self._num_channel, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2, self._num_channel * 2, 1)
                cell2_3 = cell (self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4, self._num_channel * 2, 0)


                cell3_1 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 4, self._num_channel * 4, 1)
                cell3_3 = cell (self._step, self._multiplier, self._num_channel * 4, self._num_channel * 8, self._num_channel * 4, 0)


                cell4_1 = cell (self._step, self._multiplier, self._num_channel * 8, self._num_channel * 4, self._num_channel * 8, 2)
                cell4_2 = cell (self._step, self._multiplier, self._num_channel * 8, self._num_channel * 8, self._num_channel * 8, 1)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]
        self.aspp_4 = nn.Sequential (
            ASPP (self._num_channel, 24, 24, self._num_classes)
        )

        self.aspp_8 = nn.Sequential (
            ASPP (self._num_channel * 2, 12, 12, self._num_classes)
        )
        self.aspp_16 = nn.Sequential (
            ASPP (self._num_channel * 4, 6, 6, self._num_classes)
        )
        self.aspp_32 = nn.Sequential (
            ASPP (self._num_channel * 8, 3, 3, self._num_classes)
        )
        




    def forward (self, x) :
        self.level_2 = []
        self.level_4 = []
        self.level_8 = []
        self.level_16 = []
        self.level_32 = []

        # self._init_level_arr (x)
        temp = self.stem0 (x)
        self.level_2.append (self.stem1 (temp))
        self.level_4.append (self.stem2 (self.level_2[-1]))
        weight_cells = F.softmax(self.alphas_cell, dim=-1)
        weight_network = F.softmax (self.alphas_network, dim = -1)
        count = 0
        weight_network = F.softmax (self.alphas_network, dim = -1)
        weight_cells = F.softmax(self.alphas_cell, dim=-1)
        for layer in range (self._num_layers) :

            if layer == 0 :
                level4_new = self.cells[count] (None, self.level_4[-1], weight_cells)
                count += 1
                level8_new = self.cells[count] (None, self.level_4[-1], weight_cells)
                count += 1
                self.level_4.append (level4_new * self.alphas_network[layer][0][0])
                self.level_8.append (level8_new * self.alphas_network[layer][0][1])
                # print ((self.level_4[-2]).size (),  (self.level_4[-1]).size())
            elif layer == 1 :
                level4_new_1 = self.cells[count] (self.level_4[-2], self.level_4[-1], weight_cells)
                count += 1
                level4_new_2 = self.cells[count] (self.level_4[-2], self.level_8[-1], weight_cells)
                count += 1
                level4_new = self.alphas_network[layer][0][0] * level4_new_1 + self.alphas_network[layer][0][1] * level4_new_2

                level8_new_1 = self.cells[count] (None, self.level_4[-1], weight_cells)
                count += 1
                level8_new_2 = self.cells[count] (None, self.level_8[-1], weight_cells)
                count += 1
                level8_new = self.alphas_network[layer][1][0] * level8_new_1 + self.alphas_network[layer][1][1] * level8_new_2

                level16_new = self.cells[count] (None, self.level_8[-1], weight_cells)
                level16_new = level16_new * self.alphas_network[layer][1][2]
                count += 1


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)

            elif layer == 2 :
                level4_new_1 = self.cells[count] (self.level_4[-2], self.level_4[-1], weight_cells)
                count += 1
                level4_new_2 = self.cells[count] (self.level_4[-2], self.level_8[-1], weight_cells)
                count += 1
                level4_new = self.alphas_network[layer][0][0] * level4_new_1 + self.alphas_network[layer][0][1] * level4_new_2

                level8_new_1 = self.cells[count] (self.level_8[-2], self.level_4[-1], weight_cells)
                count += 1
                level8_new_2 = self.cells[count] (self.level_8[-2], self.level_8[-1], weight_cells)
                count += 1
                # print (self.level_8[-1].size(),self.level_16[-1].size())
                level8_new_3 = self.cells[count] (self.level_8[-2], self.level_16[-1], weight_cells)
                count += 1
                level8_new = self.alphas_network[layer][1][0] * level8_new_1 + self.alphas_network[layer][1][1] * level8_new_2 + self.alphas_network[layer][1][2] * level8_new_3

                level16_new_1 = self.cells[count] (None, self.level_8[-1], weight_cells)
                count += 1
                level16_new_2 = self.cells[count] (None, self.level_16[-1], weight_cells)
                count += 1
                level16_new = self.alphas_network[layer][2][0] * level16_new_1 + self.alphas_network[layer][2][1] * level16_new_2


                level32_new = self.cells[count] (None, self.level_16[-1], weight_cells)
                level32_new = level32_new * self.alphas_network[layer][2][2]
                count += 1

                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)

            elif layer == 3 :
                level4_new_1 = self.cells[count] (self.level_4[-2], self.level_4[-1], weight_cells)
                count += 1
                level4_new_2 = self.cells[count] (self.level_4[-2], self.level_8[-1], weight_cells)
                count += 1
                level4_new = self.alphas_network[layer][0][0] * level4_new_1 + self.alphas_network[layer][0][1] * level4_new_2

                level8_new_1 = self.cells[count] (self.level_8[-2], self.level_4[-1], weight_cells)
                count += 1
                level8_new_2 = self.cells[count] (self.level_8[-2], self.level_8[-1], weight_cells)
                count += 1
                level8_new_3 = self.cells[count] (self.level_8[-2], self.level_16[-1], weight_cells)
                count += 1
                level8_new = self.alphas_network[layer][1][0] * level8_new_1 + self.alphas_network[layer][1][1] * level8_new_2 + self.alphas_network[layer][1][2] * level8_new_3

                level16_new_1 = self.cells[count] (self.level_16[-2], self.level_8[-1], weight_cells)
                count += 1
                level16_new_2 = self.cells[count] (self.level_16[-2], self.level_16[-1], weight_cells)
                count += 1
                level16_new_3 = self.cells[count] (self.level_16[-2], self.level_32[-1], weight_cells)
                count += 1
                level16_new = self.alphas_network[layer][2][0] * level16_new_1 + self.alphas_network[layer][2][1] * level16_new_2 + self.alphas_network[layer][2][2] * level16_new_3


                level32_new_1 = self.cells[count] (None, self.level_16[-1], weight_cells)
                count += 1
                level32_new_2 = self.cells[count] (None, self.level_32[-1], weight_cells)
                count += 1
                level32_new = self.alphas_network[layer][3][0] * level32_new_1 + self.alphas_network[layer][3][1] * level32_new_2


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)


            else :
                level4_new_1 = self.cells[count] (self.level_4[-2], self.level_4[-1], weight_cells)
                count += 1
                level4_new_2 = self.cells[count] (self.level_4[-2], self.level_8[-1], weight_cells)
                count += 1
                level4_new = self.alphas_network[layer][0][0] * level4_new_1 + self.alphas_network[layer][0][1] * level4_new_2

                level8_new_1 = self.cells[count] (self.level_8[-2], self.level_4[-1], weight_cells)
                count += 1
                level8_new_2 = self.cells[count] (self.level_8[-2], self.level_8[-1], weight_cells)
                count += 1
                level8_new_3 = self.cells[count] (self.level_8[-2], self.level_16[-1], weight_cells)
                count += 1
                level8_new = self.alphas_network[layer][1][0] * level8_new_1 + self.alphas_network[layer][1][1] * level8_new_2 + self.alphas_network[layer][1][2] * level8_new_3

                level16_new_1 = self.cells[count] (self.level_16[-2], self.level_8[-1], weight_cells)
                count += 1
                level16_new_2 = self.cells[count] (self.level_16[-2], self.level_16[-1], weight_cells)
                count += 1
                level16_new_3 = self.cells[count] (self.level_16[-2], self.level_32[-1], weight_cells)
                count += 1
                level16_new = self.alphas_network[layer][2][0] * level16_new_1 + self.alphas_network[layer][2][1] * level16_new_2 + self.alphas_network[layer][2][2] * level16_new_3


                level32_new_1 = self.cells[count] (self.level_32[-2], self.level_16[-1], weight_cells)
                count += 1
                level32_new_2 = self.cells[count] (self.level_32[-2], self.level_32[-1], weight_cells)
                count += 1
                level32_new = self.alphas_network[layer][3][0] * level32_new_1 + self.alphas_network[layer][3][1] * level32_new_2


                self.level_4.append (level4_new)
                self.level_8.append (level8_new)
                self.level_16.append (level16_new)
                self.level_32.append (level32_new)
        # print (self.level_4[-1].size(),self.level_8[-1].size(),self.level_16[-1].size(),self.level_32[-1].size())
        # concate_feature_map = torch.cat ([self.level_4[-1], self.level_8[-1],self.level_16[-1], self.level_32[-1]], 1)
        aspp_result_4 = self.aspp_4 (self.level_4[-1])

        aspp_result_8 = self.aspp_8 (self.level_8[-1])
        aspp_result_16 = self.aspp_16 (self.level_16[-1])
        aspp_result_32 = self.aspp_32 (self.level_32[-1])
        upsample = nn.Upsample(size=(self._crop_size,self._crop_size), mode='bilinear', align_corners=True)
        aspp_result_4 = upsample (aspp_result_4)
        aspp_result_8 = upsample (aspp_result_8)
        aspp_result_16 = upsample (aspp_result_16)
        aspp_result_32 = upsample (aspp_result_32)

        sum_feature_map1 = torch.add (aspp_result_4, aspp_result_8)
        sum_feature_map2 = torch.add (aspp_result_16, aspp_result_32)
        sum_feature_map = torch.add (sum_feature_map1, sum_feature_map2)
        return sum_feature_map


    def _initialize_alphas(self):
        k = sum(1 for i in range(self._step) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        alphas_cell = torch.tensor (1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[0], nn.Parameter(alphas_cell))

        # num_layer x num_spatial_levels x num_spatial_connections (down, level, up)
        alphas_network = torch.tensor (1e-3*torch.randn(self._num_layers, 4, 3).cuda(), requires_grad=True)
        self.register_parameter(self._arch_param_names[1], nn.Parameter(alphas_network))
        self.alphas_network_mask = torch.ones(self._num_layers, 4, 3)


    def decode_network (self) :
        best_result = []
        max_prop = 0
        def _parse (weight_network, layer, curr_value, curr_result, last) :
            nonlocal best_result
            nonlocal max_prop
            if layer == self._num_layers :
                if max_prop < curr_value :
                    # print (curr_result)
                    best_result = curr_result[:]
                    max_prop = curr_value
                return

            if layer == 0 :
                print ('begin0')
                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end0-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

            elif layer == 1 :
                print ('begin1')

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end1-1')

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()


            elif layer == 2 :
                print ('begin2')

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end2-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 2
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()
            else :

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 2
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 3
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
        network_weight = F.softmax(self.alphas_network, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse (network_weight, 0, 1, [],0)
        print (max_prop)
        return best_result

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._step):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted (range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_cell = _parse(F.softmax(self.alphas_cell, dim=-1).data.cpu().numpy())
        concat = range(2+self._step-self._multiplier, self._step+2)
        genotype = Genotype(
            cell=gene_cell, cell_concat=concat
        )

        return genotype

    def _loss (self, input, target) :
        logits = self (input)
        return self._criterion (logits, target)




def main () :
    model = AutoDeeplab (5, 12, None)
    x = torch.tensor (torch.ones (4, 3, 224, 224))
    result = model.decode_network ()
    print (result)
    print (model.genotype())
    # x = x.cuda()
    # y = model (x)
    # print (model.arch_parameters ())
    # print (y.size())

if __name__ == '__main__' :
    main ()
