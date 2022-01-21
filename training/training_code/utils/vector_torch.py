import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn

def cross_1(vector1, vector2, axis=-1, name=None):
    vector1_x, vector1_y, vector1_z = torch.unbind(vector1, axis=axis)#BxHeightxWidthx3을 vector1_x는 BxHeightxWidth,vector1_y는 BxHeightxWidth,vector1_z는 BxHeightxWidth로 unstack했다.
    vector2_x, vector2_y, vector2_z = torch.unbind(vector2, axis=axis)#BxHeightxWidthx3을 vector2_x는 BxHeightxWidth,vector2_y는 BxHeightxWidth,vector2_z는 BxHeightxWidth로 unstack했다.
    n_x = vector1_y * vector2_z - vector1_z * vector2_y#각 요소끼리 곱해서 뺀다. shape은 BxHeightxWidth이다.
    n_y = vector1_z * vector2_x - vector1_x * vector2_z#각 요소끼리 곱해서 뺀다. shape은 BxHeightxWidth이다.
    n_z = vector1_x * vector2_y - vector1_y * vector2_x#각 요소끼리 곱해서 뺀다. shape은 BxHeightxWidth이다.

    return torch.stack((n_x, n_y, n_z), axis=axis)#BxHeightxWidthx3의 크기를 가지고 있다.
