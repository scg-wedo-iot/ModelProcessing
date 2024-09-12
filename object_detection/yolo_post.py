import numpy as np
from .bbox import *

def onnx_yolo_filter_score(mat_result, score_th):
    if score_th > 0:
        arr_score = mat_result[:, 4]
        index_pass = arr_score > score_th
        mat_result = mat_result[index_pass, :]

    return mat_result

def onnx_yolo_process_classes(mat_result):
    if mat_result.size < 6:
        return mat_result
    arr_classes = np.argmax(mat_result[:, 5:], axis=1)

    mat_result_post = np.concatenate((mat_result[:, 0:5], np.expand_dims(arr_classes, 1)), axis=1)

    return mat_result_post

def onnx_yolo_extract_result(mat_result):
    mat_boxes = mat_result[:, 0:4]

    arr_score = mat_result[:, 4].flatten()

    arr_classes = mat_result[:, 5].flatten()

    return mat_boxes, arr_score, arr_classes

def onnx_yolo_maxdet(mat_result, n_maxdet=None):
    if mat_result.size < 5:
        return mat_result
    # sort by score high to low
    index_sort = mat_result[:, 4].argsort()[::-1] # high to low
    mat_result_post = mat_result[index_sort, :]

    mat_result_post = mat_result_post[0:n_maxdet, :]

    return mat_result_post

def onnx_yolo_sort_bbox(mat_result, sort_type='lrtd'):
    if mat_result.ndim == 1 or (mat_result.ndim == 2 and mat_result.shape[0] < 2):
        return mat_result

    if sort_type == 'lrtd':
        # sort axis x
        index_sort = mat_result[:, 0].argsort() # low to high
        mat_result_post = mat_result[index_sort, :]
        # sort axis y
        # index_sort = mat_result_post[:, 1].argsort() # low to high
        # mat_result_post = mat_result_post[index_sort, :]

    else:
        raise ValueError(f'not support soty_type:{sort_type}')

    return mat_result_post

def onnx_yolo_filter_class(mat_result, list_class_filter, is_postprocessed_class=True):
    if is_postprocessed_class:
        arr_classes = mat_result[:, 5]

    else:
        arr_classes = np.argmax(mat_result[:, 5:], axis=1)

    index_class_pass = np.zeros_like(arr_classes, dtype=bool)
    for class_id in list_class_filter:
        index_class_pass_in = (arr_classes == class_id)
        index_class_pass = np.logical_or(index_class_pass, index_class_pass_in)

    return mat_result[index_class_pass, :]


