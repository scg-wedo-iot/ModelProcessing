import cv2
import numpy as np

def onnx_predict(ort_sess, input_blob, create_blob=False):
    # get the name of the first input of the model
    model_input = ort_sess.get_inputs()[0]
    input_name = model_input.name
    input_shape = model_input.shape
    size_in = (input_shape[2], input_shape[3])
    if create_blob:
        input_blob = cv2.dnn.blobFromImage(
            image=input_blob.astype(np.float32),
            scalefactor=1 / 255.0,
            size=size_in,
            swapRB=True,
            crop=False
        )

    model_output = ort_sess.get_outputs()[0]
    output_name = model_output.name

    # model_meta = ort_sess.get_modelmeta()
    # dictModelClass = model_meta.custom_metadata_map['names']

    # ---------- Matric output ----------
    # onnx
    pred_onnx_run = ort_sess.run([output_name], {input_name: input_blob})
    mat_result = pred_onnx_run[0]

    return mat_result[0, :, :]