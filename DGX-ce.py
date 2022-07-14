#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
#
from mpi4py import MPI
import cupy as cp
import cupyx

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import util
import core
import dgx
import train
import mpi
import exam
import mnist

sys.setrecursionlimit(10000)

def output(path, msg):
    with open(path, 'a') as f:
        print(msg, file=f)
    #
    
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    com = MPI.COMM_WORLD
    rank = com.Get_rank()
    size = com.Get_size()
    print("(%d, %d)" % (rank, size))

    config_id = 0 # FC
    #config_id = 1 # CNN all
    #config_id = 2 # CNN separate
    #
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    train_data_batch = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_label_batch = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    batch_offset = mnist.MINI_BATCH_START[rank]
    batch_size = mnist.MINI_BATCH_SIZE[rank]
    
    cp.cuda.Device(rank).use()
    my_gpu = dgx.Dgx(rank)
    
    if config_id==0:
        r = mnist.setup_dnn(my_gpu, config_id, "./wi-fc.csv")
    else:
        return 0
    #
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
    
    wk = mpi.worker(com, rank, size, r)
    wk.mode_e = 0 # 0:ce, 1:mse
    
    if config_id==0:
        if rank==0:
            w_list = wk.train.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        else:
            w_list = []
        #
        w_list = com.bcast(w_list, root=0)

        ce = wk.evaluate()
        if rank==0:
            log = "%d, %f" % (i, ce)
            output("./log.csv", log)
            r.save_as(self, "./wi/wi-fc-0000.csv")
        #
    #
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
