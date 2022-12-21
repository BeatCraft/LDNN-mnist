#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import numpy as np

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
    #
#
import util
import core
import train
import exam
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

def output(path, msg):
    with open(path, 'a') as f:
        print(msg, file=f)
    #

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    start_time = time.time()

    config = 0 # fc
    batch_size = 60000
    mini_batch_size = 100
    batch_offset = 0
    
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config, "./wi-fc.csv")
    if r:
        pass
    else:
        return 0
    #
    r.prepare(mini_batch_size, data_size, num_class)

    train_batch_image = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_batch_label = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    
    t = train.Train(r)
    
    w_list = t.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT,])
    print(len(w_list))
    
    m = int(batch_size/mini_batch_size)
    for i in range(m):
        batch_offset = mini_batch_size*i
        r.set_batch(data_size, num_class, train_batch_image, train_batch_label, mini_batch_size, batch_offset)
        #
        t.loop_sa(i, w_list, "fc")
    #
    r.save()

    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print(("time = % s" % (t)))
    #m = int(elapsed_time/60)
    #h = int(m/60)
    #m = m - m*60
    #s = elapsed_time - h*3600 - m*60
    #print(h, m, s)
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
