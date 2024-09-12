import os.path

import tf2onnx
import onnx

def to_onnx_from_h5(model, input_shape=[1, 640, 640, 3], input_name='input', path_save=None):
    import tensorflow as tf

    input_signature = [tf.TensorSpec(input_shape, tf.float32, name=input_name)]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, path_save)

def yolo2onnx(version='v8', path_model_pt, nms=True):
    if version == 'v8':
        from ultralytics import YOLO

        # PATH_PT = sys.argv[1]
        folder_model, filename_pt = os.path.split(path_model_pt)

        filename_pure = filename_pt.split('.')[0]
        model = YOLO(path_model_pt)

        # print('export to .onnx FP32 with nms...')
        model.export(format='onnx', nms=True)
