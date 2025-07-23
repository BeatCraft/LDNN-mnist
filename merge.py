import os
import sys
import time
import numpy as np
import random
import csv
#import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
import util
import core
import exam
import train

sys.path.append(os.path.join(os.path.dirname(__file__), '../ptool'))
import tool

import mnist


#import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

value_map = {
    0: -1.0,
    1: -0.5,
    2: -0.25,
    3: -0.125,
    4: 0.0,
    5: 0.125,
    6: 0.25,
    7: 0.5,
    8: 1.0
}

WV = [-1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0]

def quantize(v):
    if v<-0.75:
        q = -1.0
        i = 0
    elif v>=-0.75 and v<=-0.375:
        q = -0.5
        i = 1
    elif v>-0.375 and v<=-0.1875:
        q = -0.25
        i = 2
    elif v>-0.1875 and v<=-0.0625:
        q = -0.125
        i = 3
    elif v>-0.0625 and v<=0.0625:
        q = 0.0
        i = 4
    elif v>0.0625 and v<=0.1875:
        q = 0.125
        i = 5
    elif v>0.1875 and v<=0.375:
        q = 0.25
        i = 6
    elif v>0.375 and v<=0.75:
        q =0.5
        i = 7
    elif v>0.75 and v<=1.0:
        q = 1.0
        i = 8
    #
    return q, i
    

def import_weight(path):
    print("import_weight(%s)" % (path))
    
    num_node_list = [256, 256, 10]
    layer_list = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for i in num_node_list:
            block = []
            for row in reader:
                line = []
                for cell in row:
                    line.append(cell)
                #
                block.append(line)
                if len(block)==i:
                    break
                #
            #
            layer_list.append(block)
        # for i
    # with
    return layer_list

def dump_len(l):
    #tabstr = ""
    #for i in range(depth):
    #    tabstr = tabstr + "\t"
    #
    size = len(l)
    print(size)
    for block in l:
        bsize = len(block)
        print("\t%d" % (bsize))
        for line in block:
            lsize = len(line)
            print("\t\t%d" % (lsize))
        #
    #

def comp(l0, l1):
    size = len(l0)
    print(size)
    
    for bi in range(size):
        bsize = len(l0[bi])
        for li in range(bsize):
            lsize = len(l0[bi][li])
            for i in range(lsize):
                #print(l0[bi][li][i], l1[bi][li][i])
                if l0[bi][li][i] != l1[bi][li][i]:
                    #print(l0[bi][li][i], l1[bi][li][i])
                    wi0 = int(l0[bi][li][i])
                    wi1 = int(l1[bi][li][i])
                    wv0 = WV[wi0]
                    wv1 = WV[wi1]
                    dif = abs(wv0 - wv1) / 2
                    if wv0>wv1:
                        nv = wv0 - dif
                    else:
                        nv = wv0 + dif
                    #
                    print(wv0, wv1, wv0+wv1/2.0, dif, nv)
                #
            #
        #
    #

def average(data_list, dsize):
    base = data_list[0]
    size = len(base)
    avg_list = []
    #print(size)
    
    for bi in range(size):
        bsize = len(base[bi])
        block = []
        for li in range(bsize):
            lsize = len(base[bi][li])
            line = []
            for i in range(lsize):
                wtotal = 0.0
                for k in range(dsize):
                    wi = int(data_list[k][bi][li][i])
                    wv = WV[wi]
                    wtotal = wtotal + wv
                #
                wavg = wtotal / dsize
                wi0 = int(base[bi][li][i])
                wv0 = WV[wi0]
                qi = quantize(wavg)
                print(wv0, ">", wavg, qi)
                line.append(qi[1])
            #
            block.append(line)
        #
        avg_list.append(block)
    #
    return avg_list

def export_weight(path, data_list):
    print("export_weight(%s)" % (path))
    with open(path, "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        for block in data_list:
            writer.writerows(block)
        # for
    # with

PATH_LIST = [
    "./wi-fc.csv.0",
    "./wi-fc.csv.1",
    "./wi-fc.csv.2",
    "./wi-fc.csv.3",
    "./wi-fc.csv.4"]

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    #if argc<3:
    #    print("error", argc)
    #
    #config = int(argvs[1])
    #mode = int(argvs[2])
    #path0 = "./wi-fc.csv.0"
    #wdata0 = import_weight(path0)
    #print(wdata)
    #dump_len(wdata0)
    #path1 = "./wi-fc.csv.1"
    #wdata1 = import_weight(path1)
    #comp(wdata0, wdata1)
    
    #print(value_map[0])
    
    data_list = []
    for p in PATH_LIST:
        wdata = import_weight(p)
        data_list.append(wdata)
    #
    
    #save_data = copy.deepcopy(wdata[0])
    #print(data_list)
    avg_list = average(data_list, len(PATH_LIST))
    print(avg_list)
    
    # save
    export_weight("./test.csv", avg_list)
    return 0
    
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
