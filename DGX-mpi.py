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
    elif config_id==1:
        r = mnist.setup_dnn(my_gpu, config_id, "./wi-cnn.csv")
    elif config_id==2:
        r = mnist.setup_dnn(my_gpu, config_id, "./wi-cnn-2.csv")
    #
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
    
    wk = mpi.worker(com, rank, size, r)
    wk.mode_e = 0 # 0:ce, 1:mse
    
    if config_id==0:
        if rank==0:
            test_batch_size = mnist.TEST_BATCH_SIZE
            test_batch_image = util.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
            test_batch_label = util.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
            #
            w_list = wk.train.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        else:
            w_list = []
        #
        w_list = com.bcast(w_list, root=0)
        
        for i in range(100):
            ce = wk.loop_sa5(i, w_list, "all")
            if rank==0:
                ac = exam.classification(r, data_size, num_class, test_batch_size, test_batch_image, test_batch_label, 1000)
                log = "%d, %d, %f, %f" % (i, batch_size, ce, ac)
                output("./log.csv", log)
                #
                r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
            #
        #  
    elif config_id==1: # CNN all
        if rank==0:
            w_list = wk.train.make_w_list([core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        else:
            w_list = []
        #
        w_list = com.bcast(w_list, root=0)
        wk.mode_e = 0 # 0:ce, 1:mse
        for idx in range(100):
            wk.loop_sa(w_list, "all", idx, 1, 50)
        #
    elif config_id==2: # CNN separate
        if rank==0:
            fc_w_list = wk.train.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        else:
            fc_w_list = []
        #
        fc_w_list = com.bcast(fc_w_list, root=0)

        if rank==0:
            cnn_w_list = wk.train.make_w_list([core.LAYER_TYPE_CONV_4])
        else:
            cnn_w_list = []
        #
        cnn_w_list = com.bcast(cnn_w_list, root=0)
            
        for idx in range(50):
            #wk.mode_w = 1
            r.propagate()
            for i in range(1, 5): # FC
                layer = r.get_layer_at(i)
                layer.lock = True
            #
            wk.loop_k(fc_w_list, "fc", idx, 1, 20)

            #wk.mode_w = 2
            for i in range(1, 5): #CNN
                layer = r.get_layer_at(i)
                layer.lock = False
            #
            wk.loop_k(cnn_w_list, "cnn", idx, 1, 10)
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
