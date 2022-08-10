#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
#
from mpi4py import MPI
#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
if sys.platform.startswith('darwin'):
    import opencl
else:
    if plat.ID==1:
        import opencl
    elif plat.ID==2:
        import dgx
        import cupy as cp
        import cupyx
    #
#
import util
import core
import train
import work
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

    config = 0 # FC
    #config = 1 # CNN all
    #config = 2 # CNN separate
    #
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    train_data_batch = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_label_batch = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    batch_offset = mnist.MINI_BATCH_START[rank]
    batch_size = mnist.MINI_BATCH_SIZE[rank]
    
    if plat.ID==0: # MBP
        return 0
    elif  plat.ID==1: # tr
        platform_id = 1
        device_id = 0
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif plat.ID==2: # nvidia
        cp.cuda.Device(rank).use()
        my_gpu = dgx.Dgx(rank)
    else:
        print("error : undefined platform")
        return 0
    #
        
    if config==0:
        r = mnist.setup_dnn(my_gpu, config, "./wi-fc.csv")
    else:
        return 0
    #
    
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
    
    wk = work.worker(com, rank, size, r)
    wk.mode_e = 0 # 0:ce, 1:mse
    
    if config==0:
        if rank==0:
            w_list = wk.train.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        else:
            w_list = []
        #
        w_list = com.bcast(w_list, root=0)
        
        for i in range(100):
            ce = wk.loop_sa5(i, w_list, "all")
            if rank==0:
                log = "%d, %s" % (i+1, '{:.10g}'.format(ce))
                output("./log.csv", log)
                spath = "./wi/wi-fc-%04d.csv" % (i+1)
                r.save_as(spath)
            #
        #  
    else:
        return 0
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
