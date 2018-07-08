import numpy as np
import tensorflow as tf
import pandas as pd
# 初始化神经网络模型所需要的环境
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
gpu_id='1'
def init_env():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
init_env()