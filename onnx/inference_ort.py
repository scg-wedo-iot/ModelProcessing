import numpy as np
import cv2
import onnxruntime as ort

def onnx_load(path_model, providers=['CUDAExecutionProvider','CPUExecutionProvider']):
#def onnx_load(path_model, providers=['CPUExecutionProvider']):
    ort_sess = ort.InferenceSession(path_model, providers=providers)

    return ort_sess

def onnx_predict(ort_sess, input_blob=None, create_blob=False, get_first_output=True):
    # get the name of the first input of the model
    model_input = ort_sess.get_inputs()[0]
    input_name = model_input.name
    input_shape = model_input.shape
    n_dim = len(input_shape)

    if n_dim == 4:
        size_in = (input_shape[2], input_shape[3])
        swapRB = True

    elif n_dim == 3:
        size_in = (input_shape[1], input_shape[2])
        swapRB = False

    else:
        raise ValueError("input_shape is not support !")

    if create_blob:
        input_blob = cv2.dnn.blobFromImage(
            image=input_blob.astype(np.float32),
            scalefactor=1 / 255.0,
            size=size_in,
            swapRB=swapRB,
            crop=False
        )

    else:
        if input_blob is None:
            raise ValueError("input_blob is None, send image")

    # output information
    model_output = ort_sess.get_outputs()[0]
    output_name = model_output.name

    # inference
    if n_dim == 3:
        input_blob = np.squeeze(input_blob, axis=0)

    pred_onnx_run = ort_sess.run([output_name], {input_name: input_blob})

    # post-process output
    if get_first_output:
        mat_result = pred_onnx_run[0]

        return mat_result[0, :, :]

    else:
        return pred_onnx_run