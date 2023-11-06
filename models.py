# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class EmbedNet(nn.Module):
    def __init__(self, config,
                 conv_kernel_size=(3, 3)):
        super(EmbedNet, self).__init__()
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.config = config
        self.window = torch.hann_window(window_length=self.config.embedding.n_embedding+2, periodic=False).cuda()
        self.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=32,
                            kernel_size=conv_kernel_size,
                            )
        self.conv2 = nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=conv_kernel_size,
                            )

        self.conv3 = nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=conv_kernel_size,
                            )

        
        self.fc1 = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.pad(x)
        x = x*self.window
        x = self.conv1(x)
        x = F.selu(x)
        x = F.max_pool2d(x, (2, 4))
        x = self.pad(x)
        x = self.conv2(x)
        x = F.selu(x)
        x = F.max_pool2d(x, (3, 4))
        x = self.pad(x)
        x = self.conv3(x)
        x = F.selu(x)
        x = F.max_pool2d(x, (2, 4))
        x_flatten = torch.flatten(x, 1)
        x_fc1 = self.fc1(x_flatten)
        x_fc1 = F.selu(x_fc1)
        output = self.fc2(x_fc1)
        output_normalized = F.normalize(output, p=2, dim=-1)
        return output_normalized