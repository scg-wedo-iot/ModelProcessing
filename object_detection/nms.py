import numpy as np
from .bbox import *
# import os
# import sys

from .yolo_post import *

def objectdet_nms(mat_result, iou_same_obj_th=0.5):
    if mat_result.size <6:
        return mat_result
    arr_classes_int = mat_result[:, 5]
    list_class_int = np.unique(mat_result[:, 5]).astype(int)

    mat_result_output = np.array([])

    for model_class in list_class_int:
        index_class = (arr_classes_int == model_class)

        mat_result_class = mat_result[index_class, :]

        while True:
            mat_boxes_class_xyxy, arr_score_class, _ = onnx_yolo_extract_result(mat_result_class)

            pos_score_class_max = np.argmax(arr_score_class)

            # log object that max score
            row_result = np.expand_dims(mat_result_class[pos_score_class_max, :], axis=0)
            if mat_result_output.size == 0:
                mat_result_output = row_result
            else:
                mat_result_output = np.concatenate((mat_result_output, row_result), axis=0)

            # box ref (max score)
            box_xyxy_class_ref = mat_boxes_class_xyxy[pos_score_class_max, :]

            # IOU between ref and other box
            iou = iou_from_bbox(box_xyxy_class_ref, mat_boxes_class_xyxy)
            
            # delete object that too overlap
            index_obj_overlap = (iou > iou_same_obj_th)
            mat_result_class = mat_result_class[~index_obj_overlap]

            if mat_result_class.size == 0:
                break

    return mat_result_output
