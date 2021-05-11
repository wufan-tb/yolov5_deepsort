import torch

from pytorch_yolov5.utils.utils import non_max_suppression


def yolov5_prediction(model,tensor_img,conf_thres,iou_thres,classes):
    with torch.no_grad():
        out=model(tensor_img)[0]
        pred = non_max_suppression(out, conf_thres, iou_thres, classes=classes)[0]
    return pred
