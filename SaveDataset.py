import os
import math
import numpy as np
import pandas as pd
import glob
import pickle
import matplotlib.pyplot as plt
from pycaret.classification import *

# ポーズごとに書き出したTraining datasetから手首始点距離も求め、
# ラベルをつけて最終のTraining datasetとする。
def readRawcoordinate(wklabel, wkPath):
    rtnList = []
    lst_dist = []

    list_of_files = glob.glob(wkPath)
    list_of_files.reverse()
    if len(list_of_files)==0: exit()

    for filename in list_of_files:
        with open(filename, 'rb') as f:
            wklst = pickle.load(f)
            rtnList.append(wklst)

            for i in range(len(wklst)):
                lst_o = list([wklst[i][0][1], 
                              wklst[i][0][2], 
                              wklst[i][0][3]])      # 手首のx,y,z
                lst_dist0 = []
                for j in range(1,21):
                    lst_t = list([wklst[i][j][1], 
                                  wklst[i][j][2], 
                                  wklst[i][j][3]])  # 手首以外の20箇所のx,y,z
                    length = math.sqrt(
                        sum((px - qx) ** 2.0 for px, qx in zip(lst_o, lst_t))
                        )
                    lst_dist0.append(length)    # 手首の座標からの20点の距離
                    # print(lst_t, length)
                # print(lst_dist0)
                lst_dist0.append(wklabel)
                lst_dist.append(lst_dist0)

            # print(lst_dist)

    return lst_dist


def maketrainingdata():

    for label in range(3):  # 0: Gu-, 1: Choki, 2: Paa-
        myPath = os.path.join(os.getcwd(), 'pos_data_{}'.format(label) , 'pos_*.dat')
        rtnList = readRawcoordinate(label, myPath)
        if label == 0:
            clmn_lst = [str(i) for i in range(1,21)]
            clmn_lst.append('cls')
            df = pd.DataFrame(data=rtnList, columns=clmn_lst)
        else:
            wkdf = pd.DataFrame(data=rtnList, columns=clmn_lst)
            df = df.append(wkdf, ignore_index=True)

    df.to_csv('gochopa.csv')


def main():
    df = pd.read_csv('gochopa.csv', index_col=0)
    df.describe()

    

if __name__ == "__main__":

    print('Start')

    maketrainingdata()

    main()

    print('End')
