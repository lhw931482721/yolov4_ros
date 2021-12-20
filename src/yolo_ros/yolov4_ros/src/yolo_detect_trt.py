import sys
import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tool.utils import *

TRT_LOGGER = trt.Logger()

def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
class Detector_trt:
    def __init__(self, engine_path,namesfile,image_size):
        # with get_engine(engine_path) as self.engine, self.engine.create_execution_context() as self.context:
        cuda.init()
        device = cuda.Device(0)
        self.ctx = device.make_context()
        self.class_names = load_class_names( namesfile)
        engine = get_engine(engine_path)
        self.context = engine.create_execution_context()
        self.image_h,self.image_w = image_size
        self.context.set_binding_shape(0, (1, 3, self.image_h,self.image_w))
        self.buffers = allocate_buffers(engine, 1)


    def realtime_detect(self, image):

        # sized = cv2.resize(image, (self.image_w,self.image_h))
        # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        # #
        # boxes = do_detect(self.model, sized, 0.5, 0.4, self.use_cuda)
        # #
        # img_plot = plot_boxes_cv2(image, boxes[0], class_names=self.class_names)
        #
        # IN_IMAGE_H, IN_IMAGE_W = image_size
        width = image.shape[1]
        height = image.shape[0]
        ta = time.time()
        # Input
        resized = cv2.resize(image, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        img_in = np.ascontiguousarray(img_in)
        # print("Shape of the network input: ", img_in.shape)
        # print(img_in)

        inputs, outputs, bindings, stream = self.buffers
        # print('Length of inputs: ', len(inputs))
        inputs[0].host = img_in
        self.ctx.push()
        trt_outputs = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        self.ctx.pop()
        # print('Len of outputs: ', len(trt_outputs))

        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, 80)

        tb = time.time()

        print('-----------------------------------')
        print('    TRT inference time: %f' % (tb - ta))
        print('-----------------------------------')

        boxes = post_processing(image, 0.5, 0.5, trt_outputs)
        # print(boxes[0])
        boxsrc = []
        for box in boxes[0]:
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            boxsrc.append([x1, y1, x2, y2, box[6]])
        # img_plot = plot_boxes_cv2(image, boxes[0], class_names=self.class_names)

        # img_plot = plot_boxes_cv2(image, boxes[0], class_names=self.class_names)
        # return boxes

        return [boxsrc]
