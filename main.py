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
    if argc==5:
        pass
    else:
        print("error in sh")
        return 0
    #
    
    config = int(argvs[1])
    mode = int(argvs[2])
    batch_size = int(argvs[3])
    idx = int(argvs[4])
    batch_offset = 0
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    
    print("config=%d" % (config))
    print("mode=%d" % (mode))
    print("batch_size=%d" % (batch_size))
    
    if plat.ID==0: # MBP
        platform_id = 0
        device_id = 1
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif  plat.ID==1: # tr
        platform_id = 1
        device_id = 0
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif plat.ID==2: # nvidia
        my_gpu = dgx.Dgx(1)
    else:
        print("error : undefined platform")
        return 0
    #
    
    if config==0:
        if mode==3: # ac
            path = "./wi/60000/256x256-100/wi-fc-%04d.csv" % (idx)
            #path = "./wi/%d/wi-fc-%04d.csv" % (idx)
            #path = "./wi/%d/wi-fc-0100.csv" % (idx)
            r = mnist.setup_dnn(my_gpu, config, path)
            print(path)
            #return 0
        else:
            r = mnist.setup_dnn(my_gpu, config, "./wi-fc.csv")
        #
    elif config==1:
        r = mnist.setup_dnn(my_gpu, config, "./wi-cnn.csv")
    elif config==2:
        r = mnist.setup_dnn(my_gpu, config, "./wi-cnn-2.csv")
    #
    if r:
        pass
    else:
        return 0
    #
    
    if mode==0: # train
        pass
    elif mode==1 or mode==3: # 1:test, 3:ac
        batch_size = mnist.TEST_BATCH_SIZE
        batch_image = util.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
        batch_label = util.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
        
        ac = exam.classification(r, data_size, num_class, batch_size, batch_image, batch_label, 1000)
        if mode==3:
            #log = "%d, %s" % (idx, '{:.10g}'.format(ce))
            msg = "%d, %f" % (idx, ac)
            output("./ac.csv", msg)
        #
        
        print(ac)
        return 0
    elif mode==2: # ec to train data : execute this onece before training
        r.prepare(batch_size, data_size, num_class)
        train_batch_image = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
        train_batch_label = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
        ce = 0.0
        n = int(mnist.TRAIN_BATCH_SIZE / batch_size)
        for i in range(n):
            r.set_batch(data_size, num_class, train_batch_image, train_batch_label, batch_size, batch_offset)
            ce += r.evaluate()
            batch_offset += batch_size
        #
        ce = ce / float(n)
        log = "%d, %s" % (0, '{:.10g}'.format(ce))
        output("./log.csv", log)
        r.save_as("./wi/wi-fc-0000.csv")
        print(ce)
        return 0
    else:
        print("error : undefined mode = %d" % (mode))
        return 0
    #
    
    print("batch_offset=%d" % (batch_offset))

    train_batch_image = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_batch_label = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
        
    #test_batch_size = mnist.TEST_BATCH_SIZE
    #test_batch_image = util.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
    #test_batch_label = util.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
        
    t = train.Train(r)
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_batch_image, train_batch_label, batch_size, 0)
    
    if config==0: # all
        w_list = t.make_w_list([core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        ce = t.loop_sa_20(0, w_list, 0)
        #for i in range(100): # 10
        #    ce = t.loop_sa5(i, w_list, "all")
        #    log = "%d, %f" % (i+1, ce)
        #    output("./log.csv", log)
        #    spath = "./wi/wi-fc-%04d.csv" % (i+1)
        #    r.save_as(spath)
        #
    elif config==2: # separate
        fc_w_list = t.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        cnn_w_list = t.make_w_list([core.LAYER_TYPE_CONV_4])
            
        for idx in range(50):
            t.mode_w = 1
            r.propagate()
            for i in range(1, 5): # FC
                layer = r.get_layer_at(i)
                layer.lock = True
            #
            t.loop_sa3(fc_w_list, "fc", idx, 1, 50)
                
            t.mode_w = 2
            for i in range(1, 5): #CNN
                layer = r.get_layer_at(i)
                layer.lock = False
            #
            t.loop_sa3(cnn_w_list, "cnn", idx, 1, 20)
        #
    else:
        print("error : undefined config = %d" % (config))
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
