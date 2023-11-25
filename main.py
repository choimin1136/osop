import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import datetime
import math
import re

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from datasets.s_coco_set import CustomDataset

import cv2
import numpy as np
from functorch.dim import dims

from rpn import ss, nms
from roi_align import roi_align
from mask import mask

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create custom dataset and dataloader
root_folder = 'datasets/'
custom_dataset = CustomDataset(root_folder, transform=transform)
# print(custom_dataset.__getitem__(1)[1].shape)
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=True)

# class #
# custom_dataset.categories

# Iterate through the dataloader


datas=[]
for inputs, annotations in dataloader:
    # Your processing logic here
    # print("hear----------------------")
    # print(inputs.shape,len(annotations))
    datas.append((inputs,annotations))
    if len(datas) >= 10:
        break


# print(type(inputs))
# print(type(annotations[0]['segmentation']))

#===============[feature map]===============================

# ResNet-50 모델 불러오기
resnet50_model = resnet50(pretrained=True)

# 4번째 레이어까지의 모델 정의
model_up_to_layer4 = torch.nn.Sequential(*list(resnet50_model.children())[:-2])

# 모델에 이미지 전달하여 특성 맵 얻기
with torch.no_grad():
    model_up_to_layer4.eval()
    features = model_up_to_layer4(inputs)

#===============[selective search]==========================module issue 해결되면 복구

import selectivesearch
def selective_search(img):
  img = np.array(img)
  _, regions = selectivesearch.selective_search(img, scale=100, min_size=2000)
  rects = [cand['rect'] for cand in regions]
  return rects

#===============[NMS]======================================module issue 해결되면 복구


def nms(boxes, iou_threshold=0.7):
    '''
    boxes(list): tuple(x, y, w, h)
    iou_threshold = 0.7 기본값
    '''

    if not boxes:
        return []

    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][2] if len(boxes[i]) > 2 else 0, reverse=True)

    keep_indices = []
    idx = []
    n = 0
    while sorted_indices:
        current_index = sorted_indices[0]
        keep_indices.append(current_index)
        current_box = boxes[current_index]
        
        remaining_indices = []

        for i in range(1, len(sorted_indices)):
            other_index = sorted_indices[i]
            other_box = boxes[other_index]
            iou = calculate_iou(current_box, other_box)
            
            if iou <= iou_threshold:
                remaining_indices.append(other_index)

        sorted_indices = remaining_indices
        selected_boxes = [boxes[i] for i in keep_indices]
        idx.append(n)
        n += 1
    return selected_boxes, idx

def calculate_iou(box1, box2):
    '''
    boxes(list): tuple(x, y, w, h)
    '''

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    area1 = w1 * h1
    area2 = w2 * h2

    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou

#===============[RoI]======================================

def ch_box_shape(boxes):
    re_box = []
    for i in boxes:
        x, y, w, h = i
        re_box.append((int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)))
    return re_box

img_ss = inputs.squeeze(0).numpy().transpose(1, 2, 0)
rects = selective_search(img_ss)
boxes, idx = nms(rects)
roi_boxes = ch_box_shape(boxes)

rois=[]
for tpl, val in zip(roi_boxes, idx):
    rois.append([val] + list(tpl))
rois_tensor = torch.as_tensor(rois, dtype=torch.float)

def bilinear_interpolate(input, x, y, width, height, xmask,  ymask):
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)

    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1. - ly
    hx = 1. - lx

    #보간에 사용되는 픽셀이 유효한지 판단 하는 함수
    def masked_index(y, x):
        y = torch.where(ymask, y, 0)
        x = torch.where(xmask, x, 0)
        return input[y, x]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4

    return val

def roi_align(input, rois, spatial_scale, pooled_width=7, pooled_height=7, sampling_ratio=-1, aligned=False):
    _, _, height, width = input.size()

    n, c, ph, pw = dims(4)
    ph.size = pooled_height
    pw.size = pooled_width
    offset_rois = rois[n]
    roi_batch_ind = offset_rois[0].int()
    offset = 0.5 if aligned else 0.0
    roi_start_w = offset_rois[1] * spatial_scale - offset
    roi_start_h = offset_rois[2] * spatial_scale - offset
    roi_end_w = offset_rois[3] * spatial_scale - offset
    roi_end_h = offset_rois[4] * spatial_scale - offset

    # h = torch.arange(pooled_height, device=input.device)
    # pw = torch.arange(pooled_width, device=input.device)
    # roi_batch_ind = input[:,0].int()
    # offset = 0.5 if aligned else 0.0
    # roi_start_w = rois[:, 1] * spatial_scale - offset  # [K]
    # roi_start_h = rois[:, 2] * spatial_scale - offset  # [K]
    # roi_end_w = rois[:, 3] * spatial_scale - offset  # [K]
    # roi_end_h = rois[:, 4] * spatial_scale - offset  # [K]

    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = torch.clamp(roi_width, min=1.0)
        roi_height = torch.clamp(roi_height, min=1.0)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    print("roi_batch_ind:", roi_batch_ind)
    print("input size:", input.size())

    offset_input = input[roi_batch_ind][0]

    roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_height / pooled_height)
    roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_width / pooled_width)

    count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)

    iy, ix = dims(2)

    iy.size = height  # < roi_bin_grid_h
    ix.size = width  # < roi_bin_grid_w

    #roi 공간에 따른 픽셀 위치 설정
    y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
    x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
    ymask = iy < roi_bin_grid_h
    xmask = ix < roi_bin_grid_w

    #쌍선형 보간 수행
    val = bilinear_interpolate(offset_input, x, y, width, height, xmask, ymask)

    #유효한 위치에 대한 마스킹 및 피쳐 맵 풀링
    val = torch.where(ymask, val, 0)
    val = torch.where(xmask, val, 0)
    output = val.sum((iy, ix))
    output /= count
    return output.order(n, c, pw, ph)
    
spatial_scale = 1.0 / 8.0
pooled_featrues = roi_align(features, rois_tensor, spatial_scale)

#====================[mask]==================================================

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

class Mask(nn.Module):
    def __init__(self, batch_size,num_rois,in_channels,pool_height,pool_weight,num_classes):
        super(Mask, self).__init__()
        self.batch_size = batch_size
        self.num_rois = num_rois
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pool_height = pool_height
        self.pool_weight = pool_weight
        self.padding = SamePad2d(kernel_size=3,stride=1)
        self.conv1 = nn.Conv2d(self.in_channels, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        # self.deconv = nn.ConvTranspose2d(256, 80, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256,self.num_classes, kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    #mask의 input은 roi aling의 output 중에, batch_size,rois,num_classes,pool_height,pool_weight만
    #input으로 받으면 된다.
    #forward의 첫번째 x는 roi_align의 아웃풋을 받는것으로, 설정해두었다.
    def forward(self, x):
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.deconv(x)
        x = self.conv2(self.padding(x))
        x = self.sigmoid(x)
        p_mask = x
        return p_mask
    