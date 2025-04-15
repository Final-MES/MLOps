import numpy as np
import pandas as pd
import os


path = "./data/raw/bosch"
date = pd.read_csv(path + "/train_date.csv")
numeric = pd.read_csv(path + "/train_numeric.csv")
category = pd.read_csv(path + "/train_categorical.csv")

