import cv2

from . pre_processing import img_preprocessing
from . yolov5_inference import yolov5_prediction
from . post_processing import detect_post_processing,track_post_processing


def Detection_Processing(input_img,save_path,yolo5_config,model,class_names,class_colors=None,area_restrict=None):
    try:
        tensor_img=img_preprocessing(input_img,yolo5_config.device,yolo5_config.img_size)
        pred=yolov5_prediction(model,tensor_img,yolo5_config.conf_thres, yolo5_config.iou_thres,yolo5_config.classes)
        result_img=detect_post_processing(input_img,pred,class_names,tensor_img.shape,class_colors,area_restrict)
        cv2.imwrite(save_path,result_img)
        return True,save_path
    except Exception as e:
        print("Wrong:",e,save_path)
        return False,e
    
def Tracking_Processing(input_img,save_path,yolo5_config,model,class_names,Tracker,class_colors=None,area_restrict=None):
    if 1:
        tensor_img=img_preprocessing(input_img,yolo5_config.device,yolo5_config.img_size)
        pred=yolov5_prediction(model,tensor_img,yolo5_config.conf_thres, yolo5_config.iou_thres,yolo5_config.classes)
        result_img=track_post_processing(input_img,pred,class_names,tensor_img.shape,Tracker,class_colors,area_restrict)
        cv2.imwrite(save_path,result_img)
        return True,save_path
    else:#except Exception as e:
        print("Wrong:",e,save_path)
        return False,e