import numpy as np
import pandas as pd
import os
import math

import Merge_RNN
import Merge_BP



def LoadBPData():
    Sheet = "Dipeptide composition (DPC) "
    Path = ""

    df = pd.read_excel(os.path.join(Path, 'BitterProtein.xlsx'), sheet_name=None)

    names = []

    # combine all feature in each sheet
    for key in list(df.keys()):
        temp = df[key].to_numpy()
        row_bias = 0
        for row in range(np.shape(temp)[0]):
            if np.isnan(temp[row - row_bias, 2]):
                temp = np.delete(temp, row - row_bias, 0)
                row_bias += 1
        column_bias = 0
        for column in range(np.shape(temp)[1]):
            if not isinstance(temp[2, column - column_bias], str):
                if np.isnan(temp[2, column - column_bias]):
                    temp = np.delete(temp, column - column_bias, 1)
                    column_bias += 1
        temp = np.delete(temp, 0, 1)
        try:
            bitter_data = np.hstack((bitter_data, temp))
        except:
            bitter_data = temp

    for i in range(bitter_data.shape[0]):
        d = df[list(df.keys())[0]].iloc[i, 0]
        names.append(d)

    df = pd.read_excel(os.path.join(Path, 'AllUmami.xlsx'), sheet_name=None)

    # umami_data=df[list(df.keys())[0]].to_numpy()
    ls = []
    exampleSheet = df[list(df.keys())[0]]
    # get threshold
    for i in range(exampleSheet.shape[0]):
        d = exampleSheet.iloc[i, 1]
        names.append(exampleSheet.iloc[i, 0])
        d2 = 0
        if type(d) is str:
            d = d.split('>')[-1].split('<')[-1].split(' ')[0].split('m')[0]
            if len(d) == 0:
                d2 = 10
            else:
                d2 = float(d)
        else:
            if math.isnan(d):
                d2 = 10
            else:
                d2 = d
        ls.append(d2)
    # combine features
    for key in list(df.keys()):
        temp = df[key].to_numpy()
        row_bias = 0
        for row in range(np.shape(temp)[0]):
            if not isinstance(temp[row - row_bias, 2], str):
                if np.isnan(temp[row - row_bias, 2]):
                    temp = np.delete(temp, row - row_bias, 0)
                    row_bias += 1
        column_bias = 0
        for column in range(np.shape(temp)[1]):
            if not isinstance(temp[2, column - column_bias], str):
                if np.isnan(temp[2, column - column_bias]):
                    temp = np.delete(temp, column - column_bias, 1)
                    column_bias += 1
        temp = np.delete(temp, [0, 1], 1)
        try:
            umami_data = np.hstack((umami_data, temp))
        except:
            umami_data = temp
    return bitter_data,umami_data,ls,names

def Preprocess(bitter_data,umami_data,ls):
    print(bitter_data.shape)
    print(umami_data.shape)

    feature_num = umami_data.shape[1]
    bitter_data = bitter_data.astype(float)
    umami_data = umami_data.astype(float)
    b3 = bitter_data.copy()
    b3[b3 == 0] = 10
    b4 = b3.min(axis=1)
    b4 = np.reshape(b4, (b4.shape[0], 1))
    b4 = np.repeat(b4, feature_num, axis=1)
    bitter_data = bitter_data / b4

    b3 = umami_data.copy()
    b3[b3 == 0] = 10
    b4 = b3.min(axis=1)
    b4 = np.reshape(b4, (b4.shape[0], 1))
    b4 = np.repeat(b4, feature_num, axis=1)
    umami_data = umami_data / b4

    data = np.concatenate((bitter_data, umami_data), axis=0).astype("float")
    label = np.append(np.ones((1, bitter_data.shape[0])) * 40, np.array(ls).reshape(1, len(ls))).astype("float")

    return data, label, feature_num, names





if __name__ == '__main__':
    print("Training RNN...")
    bitter_RNNfeature,umami_RNNfeature=Merge_RNN.RNN()

    bitter_BPdata,umami_BPdata , ls, names = LoadBPData()
    data, label, feature_num, names=Preprocess(np.hstack((bitter_BPdata,bitter_RNNfeature)),np.hstack((umami_BPdata,umami_RNNfeature)),ls)

    print("Training BP...")
    Merge_BP.Train(data, label, feature_num, names,set_epochs=2500)
