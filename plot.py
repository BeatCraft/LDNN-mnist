import os
import sys
import time

import pandas as pd
import matplotlib.pyplot as plt

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    if argc==2:
        pass
    else:
        print("error", argc)
    #
    
    csv_filename = argvs[1]
    df = pd.read_csv(csv_filename, header=None)

    plt.figure(figsize=(10, 6))
    plt.plot(df[0], df[1], label="min")#, marker='o')
    plt.plot(df[0], df[2], label="max")#, marker='x')
    plt.plot(df[0], df[3], label="avg")#, marker='s')

    plt.xlabel("Epoc")
    plt.ylabel("CE")
    #plt.title("CE vs. n")
    plt.legend()
    plt.grid(True)

    plt.show()
    
    return 0
    
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)



