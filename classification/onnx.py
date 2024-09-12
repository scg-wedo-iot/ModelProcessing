import numpy as np
import cv2

def predict(ort_sess, img):
    model_input = ort_sess.get_inputs()[0]
    input_name = model_input.name
    input_shape = model_input.shape
    input_size = (model_input.shape[1], model_input.shape[2])

    # img = np.ones(input_size, dtype=float)
    # model_result = wedo.onnx_predict(ort_sess, img, create_blob=True, input_shape=input_shape)

    input_blob = cv2.dnn.blobFromImage(
        image=img.astype(np.float32),
        scalefactor=1 / 255.0,
        size=input_size,
        swapRB=True,
        crop=False
    )
    input_blob = np.squeeze(input_blob, axis=0)

    model_output = ort_sess.get_outputs()[0]
    output_name = model_output.name

    # ---------- Matric output ----------
    # onnx
    pred_onnx_run = ort_sess.run([output_name], {input_name: input_blob})
    mat_result = pred_onnx_run[0]

    predict_class = np.argmax(mat_result)

    return