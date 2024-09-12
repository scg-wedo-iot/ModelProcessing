import cv2

def onnx_load_cv2(path_model):
    opencv_net = cv2.dnn.readNetFromONNX(path_model)

    return opencv_net

def onnx_predict_cv2(opencv_net, input_blob):
    opencv_net.setInput(input_blob)
    pred_onnx_opencv = opencv_net.forward()
    mat_result = pred_onnx_opencv[0]
    # mat_result = np.expand_dims(mat_result, axis=0)

    return mat_result