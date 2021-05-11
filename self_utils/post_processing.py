import cv2,random,os,natsort,torch
import numpy as np

from pytorch_yolov5.utils.utils import scale_coords,plot_one_box


def detect_post_processing(np_img,pred,class_names,inference_shape,class_colors=None,area_restrict=None):
    if (area_restrict is not None) and (os.path.isfile(area_restrict)):
        rail = cv2.imread(area_restrict)
        size_K=rail.shape[0]/np_img.shape[0]
        rail = cv2.cvtColor(rail, cv2.COLOR_BGR2GRAY)
        _, rail = cv2.threshold(rail, 150, 255, cv2.THRESH_BINARY)
        Area_list = np.argwhere(rail > 250).tolist()

    colors = class_colors if class_colors != None else [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(inference_shape[2:], pred[:, :4], np_img.shape).round()
        for *xyxy, conf, cls in pred:
            if (area_restrict is not None) and (os.path.isfile(area_restrict)):
                center=[int(size_K*(0.2*xyxy[1]+0.8*xyxy[3])),int(size_K*(0.5*xyxy[0]+0.5*xyxy[2]))]
                if center not in Area_list:
                    continue
            label = '%s,%.2f' % (class_names[int(cls)],conf)
            plot_one_box(xyxy, np_img, label=label, color=colors[int(cls)]) 

    if (area_restrict is not None) and (os.path.isfile(area_restrict)):
        np_img=draw_bounding(np_img,area_restrict)
    return np_img

def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def deepsort_update(Tracker,pred,inference_shape,np_img):
    pred[:, :4] = scale_coords(inference_shape[2:], pred[:, :4], np_img.shape).round()
    bbox_xywh = []
    confs = []
    labels = []
    for *xyxy, conf, cls in pred:
        img_h, img_w, _ = np_img.shape
        x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
        obj = [x_c, y_c, bbox_w, bbox_h]
        bbox_xywh.append(obj)
        confs.append([conf.item()])
        labels.append(int(cls))
    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)
    outputs = Tracker.update(xywhs, confss , labels, np_img)
    return outputs

def track_post_processing(np_img,pred,class_names,inference_shape,Tracker,class_colors=None,area_restrict=None):
    if (area_restrict is not None) and (os.path.isfile(area_restrict)):
        rail = cv2.imread(area_restrict)
        size_K=rail.shape[0]/np_img.shape[0]
        rail = cv2.cvtColor(rail, cv2.COLOR_BGR2GRAY)
        _, rail = cv2.threshold(rail, 150, 255, cv2.THRESH_BINARY)
        Area_list = np.argwhere(rail > 250).tolist()

    colors = class_colors if class_colors != None else [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]
    if pred is not None and len(pred):
        outputs=deepsort_update(Tracker,pred,inference_shape,np_img)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            labels = outputs[:, 4]
            identities = outputs[:, 5]
            for i in range(len(outputs)):
                box=bbox_xyxy[i]
                trackid=identities[i]
                label=labels[i]
                if (area_restrict is not None) and (os.path.isfile(area_restrict)):
                    center=[int(size_K*(0.2*box[1]+0.8*box[3])),int(size_K*(0.5*box[0]+0.5*box[2]))]
                    if center not in Area_list:
                        continue
                text_info = '%s,ID:%d' % (class_names[int(label)],int(trackid))
                plot_one_box(box, np_img, text_info=text_info, color=colors[int(label)]) 

    if (area_restrict is not None) and (os.path.isfile(area_restrict)):
        np_img=draw_bounding(np_img,area_restrict)
    return np_img

def draw_bounding(img,quyu_path):
    quyu = cv2.imread(quyu_path)
    quyu = cv2.resize(quyu, (img.shape[1], img.shape[0]))
    _, th = cv2.threshold(cv2.cvtColor(quyu.copy(), cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    temp=img.copy()
    temp = cv2.drawContours(temp, contours, -1, (10, 10, 250), max(3,round(0.003 * (img.shape[0] + img.shape[1]) / 2)))
    return temp

def merge_video(img_path):
    filelist = natsort.natsorted(os.listdir(img_path))
    img=cv2.imread(os.path.join(img_path,filelist[0]))
    img_size=img.shape
    fps = 25
    file_path = img_path +'_video' + ".avi"
    fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
    video = cv2.VideoWriter( file_path, fourcc, fps ,(img_size[1],img_size[0]))
    for item in filelist:
        if item.endswith('.jpg') or item.endswith('.png'):
            item = os.path.join(img_path,item)
            img = cv2.imread(item)
            video.write(img)        
    video.release()