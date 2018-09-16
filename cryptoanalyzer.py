import pandas as pd
import numpy as np
import os
import random
from sklearn import preprocessing
from collections import deque
from constants import *



def main():

    #setup the main dataframe by merging the dataframes from the dataframe from each crypto
    main_df = setUpDataFrames()

    #setup main_df with a new future and target column for training
    main_df = setUpFutureAndTarget(main_df)

    #sort the data by the time
    times = sorted(main_df.index.values)

    #the last 5 percent of data by time
    last_5pct = times[-int(0.05*len(times))]

    #the part of the main dataframe where the timestamp is in the last 5percent of time
    validation_main_df = main_df[(main_df.index >= last_5pct)]

    #the rest of main df is now everything before the last 5 percent of time
    main_df = main_df[(main_df.index < last_5pct)]

    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)

#setup dataframs for each crypto currency and then merge them into the main datafram
def setUpDataFrames():

    main_df = pd.DataFrame()
    ratios = getRatios()

    for ratio in ratios:

        df = readDataCSV(ratio)

        # rename the close and volume columns so when merged they do not conflict
        df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

        # set the index of the dataframe as the time column
        df.set_index("time", inplace=True)

        # reset dataframe to only be the close price and volume columns
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]

        if (len(main_df)) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)

    return main_df

#gets the names for each crypto currency data csv stored in crypto_data
def getRatios():
    ratiolist = []

    dirname = os.path.dirname(__file__)
    cryptodatadirectory =  os.path.join(dirname, 'crypto_data')

    directory = os.fsencode(cryptodatadirectory)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            ratiolist.append(filename.split('.csv')[0])
            continue
        else:
            continue

    return ratiolist

#read in the csv with the data for the specified crypto currency
def readDataCSV(cryptoname):
    dataframe = pd.read_csv("crypto_data/{}.csv".format(cryptoname),
                            names=["time", "low", "high", "open", "close", "volume"])


    return dataframe

#gives main_df a column for future prices and the target for each section
def setUpFutureAndTarget(main_df):
    #make a new column where the prices in the close column are shifted by 3 minutes
    main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

    #make a new column that maps the output of the classify function using the close prices and the future prices
    main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df['future']))


    return main_df

#classify a crypto as good when the price rose
def classify(current, future):

    #if future price is higher than current price
    if float(future) > float(current):
        return 1
    else:
        return 0

#preproccess the dataframe passed with normalization
def preprocess_df(df):
    #remove future column because it is no longer needed
    df = df.drop('future', 1)

    for col in df.columns:
        if col != "target":
            #normalize all the data
            df[col] = df[col].pct_change()

            #drop all 'not a number' values
            df.dropna(inplace=True)

            #scale all values [0,1]
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = setUpSequentialListData(df)

#return a list with sequences of sequential data that are associated with the 'target' for the sequence
def setUpSequentialListData(df):
    sequential_data = []

    # has items until it gets to SEQ_LEN and then starts popping items off
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        # append all data except the target
        prev_days.append([n for n in i[:-1]])

        if len(prev_days) == SEQ_LEN:
            # append with a numpy array wtih the data and the target
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    return sequential_data

if __name__ == '__main__':
    main()