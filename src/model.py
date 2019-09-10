# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ANCHORS, NUM_ANCHORS_PER_SCALE, NUM_CLASSES, NUM_ATTRIB, LAST_LAYER_DIM

Tensor = torch.Tensor


class ConvLayer(nn.Module):
    """Basic 'conv' layer, including:
     A Conv2D layer with desired channels and kernel size,
     A batch-norm layer,
     and A leakyReLu layer with neg_slope of 0.1.
     (Didn't find too much resource what neg_slope really is.
     By looking at the darknet source code, it is confirmed the neg_slope=0.1.
     Ref: https://github.com/pjreddie/darknet/blob/master/src/activations.h)
     Please note here we distinguish between Conv2D layer and Conv layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lrelu_neg_slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out


class TinyYoloLayer(nn.Module):

    def __init__(self, scale, stride):
        super(TinyYoloLayer, self).__init__()
        if scale == 'm':
            idx = (0, 1, 2)
        elif scale == 'l':
            idx = (3, 4, 5)
        else:
            idx = None
        self.anchors = torch.tensor([ANCHORS[i] for i in idx])
        self.stride = stride

    def forward(self, x):
        num_batch = x.size(0)
        num_grid = x.size(2)

        if self.training:
            output_raw = x.view(num_batch,
                                NUM_ANCHORS_PER_SCALE,
                                NUM_ATTRIB,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous().view(num_batch, -1, NUM_ATTRIB)
            return output_raw
        else:
            prediction_raw = x.view(num_batch,
                                    NUM_ANCHORS_PER_SCALE,
                                    NUM_ATTRIB,
                                    num_grid,
                                    num_grid).permute(0, 1, 3, 4, 2).contiguous()

            self.anchors = self.anchors.to(x.device).float()
            # Calculate offsets for each grid
            grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
            grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
            grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
            anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
            anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

            # Get outputs
            x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride # Center x
            y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride  # Center y
            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height
            bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
            cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, NUM_CLASSES)  # Cls pred one-hot.

            output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)
            return output


class TinyDarkNet53BackBone(nn.Module):

    def __init__(self):
        super(TinyDarkNet53BackBone, self).__init__()
        self.conv1 = ConvLayer(3, 16, 3)
        self.conv2 = ConvLayer(16, 32, 3)
        self.conv3 = ConvLayer(32, 64, 3)
        self.conv4 = ConvLayer(64, 128, 3)
        self.conv5 = ConvLayer(128, 256, 3)
        self.conv6 = ConvLayer(256, 512, 3)
        self.conv7 = ConvLayer(512, 1024, 3)
        self.maxpool_with_stride = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool_without_stride = torch.nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.maxpool_with_stride(tmp)
        tmp = self.conv2(tmp)
        tmp = self.maxpool_with_stride(tmp)
        tmp = self.conv3(tmp)
        tmp = self.maxpool_with_stride(tmp)
        tmp = self.conv4(tmp)
        tmp = self.maxpool_with_stride(tmp)
        out2 = self.conv5(tmp)
        tmp = self.maxpool_with_stride(out2)
        tmp = self.conv6(tmp)
        tmp = torch.nn.ConstantPad2d((0, 1, 0, 1), int(tmp.min()) - 1)(tmp)
        tmp = self.maxpool_without_stride(tmp)
        out1 = self.conv7(tmp)

        return out1, out2


class TinyYoloNetTail(nn.Module):
    """The tail side of the TinyYoloNet.
    It will take the result from Tiny_DarkNet53BackBone and do some upsampling and concatenation.
    It will finally output the detection result.
    Assembling Tiny_YoloNetTail and Tiny_DarkNet53BackBone will give you final result"""

    def __init__(self):
        super(TinyYoloNetTail, self).__init__()

        self.detect1 = TinyYoloLayer('l', 32)
        self.detect2 = TinyYoloLayer('m', 16)
        self.conv1 = ConvLayer(1024, 256, 1)
        self.conv2 = ConvLayer(256, 512, 3)
        self.conv3 = nn.Conv2d(512, NUM_ANCHORS_PER_SCALE * (4 + 1 + NUM_CLASSES), 1, bias=True, padding=0)
        self.conv4 = ConvLayer(256, 128, 1)
        self.conv5 = ConvLayer(384, 256, 3)
        self.conv6 = nn.Conv2d(256, NUM_ANCHORS_PER_SCALE * (4 + 1 + NUM_CLASSES), 1, bias=True, padding=0)

    def forward(self, x1, x2, training):

        branch = self.conv1(x1)
        tmp = self.conv2(branch)
        tmp = self.conv3(tmp)
        out1 = self.detect1(tmp)
        tmp = self.conv4(branch)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, x2), 1)
        tmp = self.conv5(tmp)
        tmp = self.conv6(tmp)
        out2 = self.detect2(tmp)

        return out1, out2


class TinyYoloNetV3(nn.Module):

    def __init__(self, nms=False, post=True):
        super(TinyYoloNetV3, self).__init__()
        self.darknet = TinyDarkNet53BackBone()
        self.tiny_yolo_tail = TinyYoloNetTail()
        self.nms = nms
        self._post_process = post

    def forward(self, x):
        tmp1, tmp2 = self.darknet(x)
        out1, out2 = self.tiny_yolo_tail(tmp1, tmp2)
        out = torch.cat((out1, out2), 1)
        logging.debug("The dimension of the output before nms is {}".format(out.size()))
        return out

    def tiny_yolo_last_layers(self):
        _layers = [self.tiny_yolo_tail.conv6,
                   self.tiny_yolo_tail.conv3]
        return _layers

    def tiny_yolo_last_two_layers(self):
        _layers = self.yolo_last_layers() + \
                  [self.tiny_yolo_tail.conv5,
                   self.tiny_yolo_tail.conv2]
        return _layers

    def tiny_yolo_last_three_layers(self):
        _layers = self.yolo_last_two_layers() + \
                  [self.tiny_yolo_tail.conv4,
                   self.tiny_yolo_tail.conv1]
        return _layers

    def tiny_yolo_tail_layers(self):
        _layers = [self.yolo_tail]
        return _layers

    def tiny_yolo_last_n_layers(self, n):
        try:
            n = int(n)
        except ValueError:
            pass
        if n == 1:
            return self.tiny_yolo_last_layers()
        elif n == 2:
            return self.tiny_yolo_last_two_layers()
        elif n == 3:
            return self.tiny_yolo_last_three_layers()
        elif n == 'tail':
            return self.tiny_yolo_tail_layers()
        else:
            raise ValueError("n>3 not defined")
