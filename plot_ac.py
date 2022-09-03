#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import math
import csv
import statistics

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
import util
import core

def csv_to_list(path):
    data_list = []
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                line = []
                for cell in row:
                    line.append(cell)
                #
                data_list.append(line)
            #
        #
        return data_list
    except Exception as e:
        print("error: %s" % e)
        return data_list
    #
    return data_list

def main():
    argvs = sys.argv
    argc = len(argvs)
    #print(argc)
    #print(argvs)
    
    pdir = "./"
    ac_path1 = pdir + "ac-200.csv"
    ac_path2 = pdir + "ac-150.csv"
    ac_path3 = pdir + "ac-125.csv"
    ac_path4 = pdir + "ac-110.csv"
    #ce_path = pdir + "log-256-20.csv"
    #ce_list = csv_to_list(ce_path)
    #ac_path = pdir + "ac-256-20.csv"
    ac_list1 = csv_to_list(ac_path1)
    ac_list2 = csv_to_list(ac_path2)
    ac_list3 = csv_to_list(ac_path3)
    ac_list4 = csv_to_list(ac_path4)
    
    #data_list = csv_to_list("./batch_ac.csv")
    dsize = len(ac_list1)
    print(len(ac_list1))
    print(len(ac_list2))
    print(len(ac_list3))
    print(len(ac_list4))
        
    xlist = []
    y0_list = []
    y1_list = []
    y2_list = []
    y3_list = []
    labels = []
    for i in range(dsize):
        xlist.append(i)
        #ce_list[i])
        #labels.append(d[0])
        y0_list.append(float(ac_list1[i][1]))
        y1_list.append(float(ac_list2[i][1]))
        y2_list.append(float(ac_list3[i][1]))
        y3_list.append(float(ac_list4[i][1]))
    #
    #print(xlist)
    #print(ylist)

    plt.figure(figsize=(4,2))
    #plt.figure(figsize=(10,4))
    #plt.title('Accuracy Changes in Batch Sizes', loc='center')
    #plt.xlabel("Size of Batch") # , size = 16
    #plt.xlim()
    plt.ylim(0.9, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.plot(xlist, y0_list, label="2.00", lw=0.4)
    plt.plot(xlist, y1_list, label="1.50", lw=0.4)
    plt.plot(xlist, y2_list, label="1.25", lw=0.4)
    plt.plot(xlist, y3_list, label="1.10", lw=0.4)
    
    
    
    plt.legend(loc='lower right', fontsize=6)
    #plt.plot(x, y2, label="cos x")
    #plt.xlim(0, 600)
    
    #plt.xticks(xlist, rotation=90)
    #plt.xticks([100, 1000, 5000, 10000, 20000, 40000, 60000], rotation=90)
    #plt.xticks([1000, 10000, 20000, 40000, 60000])
    #plt.ylim(0.6, 1.0)
    plt.tight_layout()
    plt.savefig("./fig.png", format="png", dpi=300)
    #plt.show()
    
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
