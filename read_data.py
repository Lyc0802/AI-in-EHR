import numpy as np
import pandas as pd
import os

def read_data(file_path):
    """
    Reads data from a CSV file and returns the features and labels.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing the features (X) and labels (y).
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the CSV file
    data = pd.read_csv(file_path)

    # Check if the data is empty
    if data.empty:
        raise ValueError("The CSV file is empty.")

    return data

data = read_data('data\icu\_chartevents.csv')
# Display the first few rows of the data
print(data.head())