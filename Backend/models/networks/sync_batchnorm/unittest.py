# -*- coding: utf-8 -*-
# File   : unittest.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import unittest
import torch


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, x, y):
        adiff = float((x - y).abs().max())
        rdiff = 'NaN' if (y == 0).all() else float((adiff / y).abs().max())
        message = f'Tensor close check failed\nadiff={adiff}\nrdiff={rdiff}\n'
        self.assertTrue(torch.allclose(x, y), message)

