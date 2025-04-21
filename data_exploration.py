"""
Exploratory data analysis for project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# change working directory
os.chdir(r"C:\Users\mkeo9\OneDrive\NEU\DS 5220 Supervised Machine Learning\Project")

# load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

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
df.to_csv("cleaned_news.csv", index=False)

