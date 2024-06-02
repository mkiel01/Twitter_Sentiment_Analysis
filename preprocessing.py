import numpy as np
import pandas as pd

import os

import pandas as pd

def load_data(path):
    path = '/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/training.1600000.processed.noemoticon.csv'
    data = pd.read_csv(path, encoding='ISO-8859-1')
    return data

def show_head(data, message):
    print(f"data head 5: {message} \n")
    print(data.head(5))



def columns_renaming(data):
    data = data.drop(data.columns[[1,3,4]], axis=1)
    data.columns.values[0] = "sentiment"
    data.columns.values[1] = "date"
    data.columns.values[2] = "tweet"
    return data 

def change_size(data_size_percentage, original_data, random_seed=42):
    print("size before change: \n")
    print(len(original_data))
    data = original_data.sample(frac=data_size_percentage / 100, random_state=random_seed)
    print(f"size after change {data_size_percentage}%: \n")    
    print(len(data))
    return data 

def save_data_to_csv(data, path):
    data.to_csv(path)
    print(f"data succesfully saved to csv in path: {path}\n")


def assign_new_colum(data, name):
     data[name] = -1
     return data

def calc_sentiment(data):
    # Initialize counters
    neg = 0
    neut = 0
    pos = 0

    # Loop through the DataFrame
    for i in range(len(data)):
        if data["sentiment"][i] == "0":
            neg += 1
        elif data["sentiment"][i] == "2":
            neut += 1
        elif data["sentiment"][i] == "4":
            pos += 1

    print(f"pos: {pos} \n")
    print(f"neut: {neut} \n")
    print(f"neg: {neg} \n")
    return neg, neut, pos

            
