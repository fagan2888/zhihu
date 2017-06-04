# -*- coding: utf-8 -*-

import gc

import numpy as np
import pandas as pd

import config
from utils import pkl_utils

def main():
	dfTrain = pd.read_csv(config.TRAIN_DATA)
	print dfTrain.head(10)

if __name__ == "__main__":
	main()