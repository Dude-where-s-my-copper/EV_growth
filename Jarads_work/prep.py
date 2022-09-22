import pandas as pd
import numpy as np

def time_series_split(df):
    train_size = .90
    n = df.shape[0]
    test_start_index = round(train_size * n)

    train = df[:test_start_index] # everything up (not including) to the test_start_index
    test = df[test_start_index:]

    train_size = .90
    n2 = train.shape[0]
    validate_start_index = round(train_size * n2)

    train = df[:validate_start_index] # everything up (not including) to the test_start_index
    validate = df[validate_start_index:test_start_index ]

    return train, validate, test