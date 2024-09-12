import numpy as np

def box2RowCol(boxes,w,h):
    # img_shape = img.shape

    # h = img.shape[0]
    # w = img.shape[1]
    # position of box in range 0 - 1
    ymin = boxes[0]
    xmin = boxes[1]
    ymax = boxes[2]
    xmax = boxes[3]

    # Convert range 0 - 1 to pixel in range image width and height
    xmin = int(xmin * w)
    xmax = int(xmax * w)
    ymin = int(ymin * h)
    ymax = int(ymax * h)

    return ymin, xmin, ymax, xmax

def boxes2point(boxes, format_boxes):

    if format_boxes == 'row_col':
        y_str = boxes[0]
        y_end = boxes[1]
        x_str = boxes[2]
        x_end = boxes[3]

    elif format_boxes == 'xyxy':
        x_str = int(boxes[0])
        y_str = int(boxes[1])
        x_end = int(boxes[2])
        y_end = int(boxes[3])

    elif format_boxes == 'xywh':
        x_c = int(boxes[0])
        y_c = int(boxes[1])
        w = int(boxes[2])
        h = int(boxes[3])

        offset_w = int(w/2)
        offset_h = int(h/2)

        x_str = x_c - offset_w
        y_str = y_c - offset_h
        x_end = x_c + offset_w
        y_end = y_c + offset_h

    start_point = (x_str, y_str)
    end_point = (x_end, y_end)

    return start_point, end_point

def convertBox(mat_box, input_type, output_type):
    if mat_box.size < 4:
        return mat_box

    mat_box_converted = mat_box.copy()
    if input_type == output_type:
        return mat_box_converted

    if input_type == "xywh":
        if output_type == "xyxy":
            mat_box_converted[:, 0] = mat_box[:, 0] - mat_box[:, 2]/2
            mat_box_converted[:, 1] = mat_box[:, 1] - mat_box[:, 3]/2
            mat_box_converted[:, 2] = mat_box[:, 0] + mat_box[:, 2]/2
            mat_box_converted[:, 3] = mat_box[:, 1] + mat_box[:, 3]/2

    elif input_type == 'xyxy':
        if output_type == 'xywh':
            # x_center, y_center, w, h
            w = mat_box[:, 2] - mat_box[:, 0]
            h = mat_box[:, 3] - mat_box[:, 1]
            mat_box_converted[:, 0] = (mat_box[:, 0] + mat_box[:, 2])/2
            mat_box_converted[:, 1] = (mat_box[:, 1] + mat_box[:, 3])/2
            mat_box_converted[:, 2] = w
            mat_box_converted[:, 3] = h

    # if np.any(mat_box_converted > 640):
    #     pass

    return mat_box_converted

def bbox2area(mat_boxes, bbox_type='xyxy'):
    if bbox_type == 'xyxy':
        if mat_boxes.ndim == 1:
            area = (mat_boxes[2] - mat_boxes[0]) * (mat_boxes[3] - mat_boxes[1])
        elif mat_boxes.ndim == 2:
            area = (mat_boxes[:, 2] - mat_boxes[:, 0]) * (mat_boxes[:, 3] - mat_boxes[:, 1])

    elif bbox_type == 'xywh':
        if mat_boxes.ndim == 1:
            area = mat_boxes[2] * mat_boxes[3]
        elif mat_boxes.ndim == 2:
            area = mat_boxes[:, 2] * mat_boxes[:, 3]

    else:
        raise ValueError(f"Not support bbox_type:{bbox_type}")

    return area

def iou_from_bbox(bbox_xyxy_ref, mat_bbox_xyxy_other):
    mat_bbox_xyxy_ref = np.ones_like(mat_bbox_xyxy_other) * bbox_xyxy_ref.flatten()

    # find w, h of intersect
    x_i_min = np.max(np.column_stack((mat_bbox_xyxy_ref[:, 0], mat_bbox_xyxy_other[:, 0])), axis=1)
    x_i_max = np.min(np.column_stack((mat_bbox_xyxy_ref[:, 2], mat_bbox_xyxy_other[:, 2])), axis=1)
    w_i = (x_i_max - x_i_min)
    w_i[w_i < 0] = 0

    y_i_min = np.max(np.column_stack((mat_bbox_xyxy_ref[:, 1], mat_bbox_xyxy_other[:, 1])), axis=1)
    y_i_max = np.min(np.column_stack((mat_bbox_xyxy_ref[:, 3], mat_bbox_xyxy_other[:, 3])), axis=1)
    h_i = (y_i_max - y_i_min)
    h_i[h_i < 0] = 0

    # all
    area_class_ref = np.ones((mat_bbox_xyxy_other.shape[0])) * bbox2area(bbox_xyxy_ref, 'xyxy')
    area_class_other = bbox2area(mat_bbox_xyxy_other, 'xyxy')
    area_class_intersect = w_i * h_i
    iou = area_class_intersect / (area_class_ref + area_class_other - area_class_intersect)

    if (iou > 1).any():
        pass

    return iou

def rescale_bbox(mat_result, img_size_original=None, img_size_model=(640, 640)):
    if mat_result.size < 4:
        return mat_result

    mat_result_post = mat_result.copy()
    if img_size_original[0] != img_size_model[0]:
        mat_result_post[:, 0] = (mat_result[:, 0] / img_size_model[0]) * img_size_original[0]
        mat_result_post[:, 2] = (mat_result[:, 2] / img_size_model[0]) * img_size_original[0]

    if img_size_original[1] != img_size_model[1]:
        mat_result_post[:, 1] = (mat_result[:, 1] / img_size_model[1]) * img_size_original[1]
        mat_result_post[:, 3] = (mat_result[:, 3] / img_size_model[1]) * img_size_original[1]

    return mat_result_post

def point2rowcol(start_point, end_point):
    index_row = [0, 0]
    index_row[0] = start_point[1]
    index_row[1] = end_point[1]

    index_col = [0, 0]
    index_col[0] = start_point[0]
    index_col[1] = end_point[0]

    return index_row, index_col

def bbox2rowcol(bbox, bbox_type):
    start_point, end_point = boxes2point(bbox, bbox_type)
    index_row, index_col = point2rowcol(start_point, end_point)

    return index_row, index_col

def crop_from_bbox(img, bbox, bbox_type=None):
    index_row, index_col = bbox2rowcol(bbox, bbox_type)

    img_crop = img[index_row[0]:index_row[1], index_col[0]:index_col[1], :]

    return img_crop
