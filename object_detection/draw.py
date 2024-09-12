import cv2
from .bbox import *

def drawMultiRectangle(img, boxes, format_boxes = 'row_col', color = (0,0,255), thickness = 1,
                       label=None, offset_label = (0, -5)):
    '''
        color = BGR
    '''
    imgOut = img.copy()
    box_ndim = boxes.ndim
    if box_ndim == 2:
        nBoxes = boxes.shape[0]

    elif box_ndim == 1:
        nBoxes = 1

    for ibox in range(nBoxes):
        if nBoxes > 1 or box_ndim > 1:
            box = boxes[ibox, :]
        elif nBoxes == 1:
            box = boxes
        start_point, end_point = boxes2point(box, format_boxes)

        # imgOut = cv2.rectangle(imgOut, (list_point[0], list_point[1]), (list_point[2], list_point[3]),
        #                         color=color, thickness=thickness)
        imgOut = cv2.rectangle(imgOut, start_point, end_point,
                                color=color, thickness=thickness)

        if not label is None:
            text = str(int(label[ibox]))
            x_label = start_point[0] + offset_label[0]
            y_label = start_point[1] - thickness + offset_label[1]
            start_point_text = (x_label, y_label)
            imgOut = cv2.putText(imgOut, text, start_point_text, cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                 color, 1, cv2.LINE_AA)

    return imgOut
