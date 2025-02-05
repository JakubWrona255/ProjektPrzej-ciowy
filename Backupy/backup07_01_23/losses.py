from imports import *


def complete_IoU_cofidence_loss(y_true, y_pred):
    return iou_loss(y_true, y_pred) + confidence_loss(y_true, y_pred)


def confidence_loss(y_true, y_pred):
    diff = K.transpose(y_true)[0] - K.transpose(y_pred)[0]
    return diff**2


def iou_loss(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1) * K.abs(
        K.transpose(y_true)[4] - K.transpose(y_true)[2] + 1)

    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1) * K.abs(
        K.transpose(y_pred)[4] - K.transpose(y_pred)[2] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_1 = K.maximum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_2 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])
    overlap_3 = K.minimum(K.transpose(y_true)[4], K.transpose(y_pred)[4])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_los = -K.log(iou)

    return iou_los
