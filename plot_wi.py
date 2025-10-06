#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import pickle
import numpy as np
import csv
import random
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#WEIGHT_SET = [-1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0]
WEIGHT_SET_0 = [-1.0, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1.0] # 11
WEIGHT_SET_1 = [-1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0] # 9
WEIGHT_SET_2 = [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0] # 8
WEIGHT_SET_3 = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
 0,
 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 21
 
str_WEIGHT_SET_0 = ["-1.0", "-0.5", "-0.25", "-0.125", "-0.0625", "0", "0.0625", "0.125", "0.25", "0.5", "1.0"]
str_WEIGHT_SET_1 = ["-1.0", "-0.5", "-0.25", "-0.125", "0", "0.125", "0.25", "0.5", "1.0"]
str_WEIGHT_SET_2 = ["-1.0", "-0.5", "-0.25", "-0.125", "0.125", "0.25", "0.5", "1.0"]
str_WEIGHT_SET_3 = ["-1.0", "-0.9", "-0.8", "-0.7", "-0.6", "-0.5", "-0.4", "-0.3", "-0.2", "-0.1",
 "0",
 "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"] # 21
 
def import_synapse_network(path):
    print("import_synapse_network(%s)" % (path))
    with open(path, "r") as f:
            reader = csv.reader(f)
            wi_list = []
            for row in reader:
                li = int(row[0])
                ri = int(row[1])
                wi = int(row[2])
                wv = float(row[3])
                wi_list.append((wi, wv))
            #
            #print(len(wi_list))
            return wi_list
    #
    return None

def import_weight_index(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        i = 0
        data_list = []
        print(type(reader))
        for row in reader:
            for cell in row:
                data_list.append(int(cell))
            #
        #
        return data_list
    #
    return None


def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    
    if argc!=2:
        return 0
    #
    wi_path = argvs[1]
    wv_list = import_weight_index(wi_path)
    #print(wv_list)
    
    num_wi = len(WEIGHT_SET_3)
    hist_list = [0]*num_wi
    print(hist_list, len(wv_list))
        
    for wi in wv_list:
        hist_list[wi] = hist_list[wi] + 1
    #
    print(hist_list)
    for h in hist_list:
        print(h/len(wv_list))
    #
        
    plt.figure(figsize=(4,2))
    #plt.plot(WEIGHT_SET, hist_list)
    #plt.bar([0,1,2,3,4,5,6,7,8], hist_list, linewidth=0)
    plt.bar(str_WEIGHT_SET_3, hist_list, width=0.5, linewidth=0)
    
    #plt.plot(WEIGHT_SET, hist_list, label="Weight")
    #plt.plot(xlist, y1_list, label="Accuracy")
    #plt.legend()
    plt.tight_layout()
    #plt.savefig("./fig.png", format="png", dpi=300)
    plt.show()
    
    return 0


if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
