# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import caffe

MODEL_FILE = './lenet_deploy.prototxt'
PRETRAINED = './lenet_iter_20000.caffemodel'

argvs = sys.argv
argc = len(argvs)

if argc < 2:
    print("Usage: pyrhon classfy_chars.py <imagepath>")
    sys.exit(1)

IMAGE_FILE = argvs[1]

if not os.path.isfile(PRETRAINED):
    print("error: pre-trained cafe model is not found")

caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED, channel_swap=[0], image_dims=(28, 28))

input_image = caffe.io.load_image(IMAGE_FILE, color=False)

prediction = net.predict([input_image], False)

print("prediction shape: {}" .format(prediction[0].shape))
print("predicted class: {}" .format(prediction[0].argmax()))
