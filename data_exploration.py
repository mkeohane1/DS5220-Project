"""
Exploratory data analysis for project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load data
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# label data
fake["label"] = 0
real["label"] = 1

# combine datasets
df = pd.concat([fake, real]).reset_index(drop=True)

# check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicates in combined dataset: {duplicates}")

# check shape of datasets (# of rows and columns)
print("Fake news rows, columns:", fake.shape)
print("Real news rows, columns:", real.shape)

# check for missing values in each dataset
print("Missing values in fake dataset:\n", fake.isnull().sum())
print("Missing values in real dataset:\n", real.isnull().sum())

# check the first few rows of each dataset
print(f"Fake news:", fake.head())
print(f"Real news:", real.head())

# save combined dataset to a new CSV file
df.to_csv("data/cleaned_news.csv", index=False)

