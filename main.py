#! /usr/bin/python
# -*- coding: utf-8 -*-
#

import os
import sys
import time
#from stat import *
import numpy as np
#from PIL import Image

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import util
import core
import train
import exam

if sys.platform.startswith('darwin'):
    import opencl
else:
    import dgx
#

import mnist

sys.setrecursionlimit(10000)
       
def save_array_to_png(array, w, h, path): # uint8, 1d, RGB
    data_in = np.reshape(array, (3, w*h))
    r = data_in[0]
    g = data_in[1]
    b = data_in[2]
    
    data = np.zeros((w*h, 3), dtype=np.uint8)
    
    for i in range(w*h):
        data[i][0] = r[i]
        data[i][1] = g[i]
        data[i][2] = b[i]
    #
    data = np.reshape(data, (h, w, 3)) # (2048, 1536, 4)
    pimg = Image.fromarray(data)
    pimg.save(path)

def save_array_gray_to_png(array, w, h, path): # uint8, 1d
    data = np.reshape(array, (h, w))
    pimg = Image.fromarray(data)
    pimg.save(path)

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    if argc==7:
        pass
    else:
        print("error in sh")
        return 0
    #
    
    type_id = int(argvs[1])
    platform_id = int(argvs[2])
    device_id = int(argvs[3])
    config = int(argvs[4])
    mode = int(argvs[5])
    batch_size = int(argvs[6])
    batch_offset = 0
    
    print("type_id=%d" % (type_id))
    print("platform_id=%d" % (platform_id))
    print("device_id=%d" % (device_id))
    print("config=%d" % (config))
    print("mode=%d" % (mode))
    print("batch_size=%d" % (batch_size))
    
    if type_id==0:
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif type_id==1:
        my_gpu = dgx.Dgx(1)
    else:
        print("error : undefined type =%d" % (type_id))
        return 0
    #
    
    r = mnist.setup_dnn(my_gpu, config)
    if r:
        pass
    else:
        return 0
    #
    
    #r.set_scale_input(1)
    #r.set_path("./wi.csv")
    #r.load()
    #r.update_weight()

    if mode==0: # train
        print("batch_offset=%d" % (batch_offset))
        
        batch_image = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
        batch_label = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
        data_size = mnist.IMAGE_SIZE
        num_class = mnist.NUM_CLASS
        batch_offset = 0
        #print(batch_image[0])
        t = train.Train(r)
        r.prepare(batch_size, data_size, num_class)
        r.set_batch(data_size, num_class, batch_image, batch_label, batch_size, batch_offset)
        t.mode = 0
        t.mse_idx = 4

        #ce = r.evaluate(1)
        #print(ce)
        
        #layer = r.get_layer_at(2)
        #if layer:
        #    mse = layer.mse()
        #    print(mse)
        #
        t.loop(50)
    elif mode==1: # test
        batch_size = mnist.TEST_BATCH_SIZE
        batch_image = util.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
        batch_label = util.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
        data_size = mnist.IMAGE_SIZE
        num_class = mnist.NUM_CLASS
        
        exam.classification(r, data_size, num_class, batch_size, batch_image, batch_label, 1000)
    else:
        print("error : undefined mode = %d" % (mode))
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
