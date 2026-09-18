"""ResNet1D for IMU regression — structure follows RoNIN's model_resnet1d.py,
sized down for 10 Hz data and a scalar target.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def conv3(cin, cout, stride=1, dilation=1):
    return nn.Conv1d(cin, cout, 3, stride=stride, padding=dilation,
                     dilation=dilation, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, down=None):
        super().__init__()
        self.c1, self.b1 = conv3(cin, cout, stride), nn.BatchNorm1d(cout)
        self.c2, self.b2 = conv3(cout, cout), nn.BatchNorm1d(cout)
        self.relu, self.down = nn.ReLU(inplace=True), down

    def forward(self, x):
        idt = x
        o = self.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        if self.down is not None:
            idt = self.down(x)
        return self.relu(o + idt)


class ResNet1D(nn.Module):
    def __init__(self, in_ch=6, out_dim=1, base=32, layers=(2, 2, 2, 2)):
        super().__init__()
        self.inp = base
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        self.l1 = self._make(base, layers[0])
        self.l2 = self._make(base * 2, layers[1], 2)
        self.l3 = self._make(base * 4, layers[2], 2)
        self.l4 = self._make(base * 8, layers[3], 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(base * 8, 128), nn.ReLU(inplace=True),
            nn.Linear(128, out_dim))

    def _make(self, cout, n, stride=1):
        down = None
        if stride != 1 or self.inp != cout:
            down = nn.Sequential(nn.Conv1d(self.inp, cout, 1, stride, bias=False),
                                 nn.BatchNorm1d(cout))
        layers = [BasicBlock1D(self.inp, cout, stride, down)]
        self.inp = cout
        layers += [BasicBlock1D(cout, cout) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):                       # x: (B, C, T)
        x = self.stem(x)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        return self.head(self.pool(x))
