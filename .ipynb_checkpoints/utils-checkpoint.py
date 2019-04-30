import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL
import argparse

def int_tuple(s):
  return tuple(int(i) for i in s.split(','))

def bool_flag(s):
  if s == '1':
    return True
  elif s == '0':
    return False
  msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
  raise ValueError(msg % s)
