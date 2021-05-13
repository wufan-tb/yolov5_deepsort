import torch,sys,argparse,cv2,random,os,time,subprocess,natsort
import numpy as np
from datetime import datetime
from multiprocessing import Pool

from self_utils.multi_tasks import Detection_Processing,Tracking_Processing,Denseing_Processing,Counting_Processing
from self_utils.pre_processing import Image_Capture
from self_utils.post_processing import merge_video,Area_Restrict,Count_Line,Object_Counter

sys.path.append('pytorch_yolov5/')
from deep_sort.configs.parser import get_config
from deep_sort.deep_sort import DeepSort

def main(yolo5_config):
    print("=> main task started: {}".format(datetime.now().strftime('%H:%M:%S')))
    
    # * load model
    a=time.time()
    Model=torch.load(yolo5_config.weights,map_location=lambda storage, loc: storage.cuda(int(yolo5_config.device)))['model'].float().fuse().eval()
    class_names = Model.module.names if hasattr(Model, 'module') else Model.names
    class_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]
    b=time.time()
    print("=> load model, cost:{:.2f}s".format(b-a))
    
    # * clean output folder
    sys_cmd="rm -rf {}".format(yolo5_config.output)
    child = subprocess.Popen(sys_cmd,shell=True)
    child.wait()
    os.makedirs(yolo5_config.output,exist_ok=True)
    c=time.time()
    print("=> clean the output path, cost:{:.2f}s".format(c-b))
    
    # * multi process
    if yolo5_config.pools > 1: 
        myP = Pool(yolo5_config.pools)
        print("=> using process pool")
    else:
        print("=> using single process")
        
    # * init deepsort tracker
    if yolo5_config.task in ['track','count']:
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        deepsort_tracker = DeepSort(cfg.DEEPSORT.REID_CKPT, max_dist=cfg.DEEPSORT.MAX_DIST, 
                            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE, 
                            n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # * load image and process
    mycap=Image_Capture(yolo5_config.input)
    if yolo5_config.task=='count':
        theLine=Count_Line([600,50],[600,1700])
        class_list=yolo5_config.classes if yolo5_config.classes is not None else [0,1,2,3]
        Obj_Counter=Object_Counter([class_names[key] for key in class_list])
    else:
        cameArea=Area_Restrict(yolo5_config.area,[mycap.get_height(),mycap.get_width()])
    total_num=mycap.get_length()
    while mycap.ifcontinue():
        ret,img,img_name = mycap.read()
        if ret:
            save_path=os.path.join(yolo5_config.output,img_name)
            if yolo5_config.pools >1:
                assert yolo5_config.task in ['detect','dense'], "!!! warning! task with track can not use multi process"
                if yolo5_config.task=='detect':
                    myP.apply_async(Detection_Processing, args=(img,save_path,yolo5_config,Model,class_names,cameArea,class_colors,))
                if yolo5_config.task=='dense':
                    myP.apply_async(Denseing_Processing,args=(img,save_path,yolo5_config,Model,class_names,cameArea,class_colors,))
            else:
                if yolo5_config.task=='empty':
                    cv2.imwrite(save_path,img)
                    time.sleep(1)
                if yolo5_config.task=='detect':
                    Detection_Processing(img,save_path,yolo5_config,Model,class_names,cameArea,class_colors)
                if yolo5_config.task=='track':
                    Tracking_Processing(img,save_path,yolo5_config,Model,class_names,cameArea,deepsort_tracker,class_colors)
                if yolo5_config.task=='dense':
                    Denseing_Processing(img,save_path,yolo5_config,Model,class_names,cameArea,class_colors)
                if yolo5_config.task=='count':
                    Counting_Processing(img,save_path,yolo5_config,Model,class_names,theLine,deepsort_tracker,Obj_Counter,class_colors)
        sys.stdout.write("\r=> processing at %d; total: %d" %(mycap.get_index(), total_num))    
        sys.stdout.flush()

    if yolo5_config.pools > 1:
        myP.close()
        myP.join()
    mycap.release()
    print("\n=> {}张图片，共花费{:.2f}s，成功{}张".format(total_num,time.time()-c,len(os.listdir(yolo5_config.output))))
    
    # * merge video
    if yolo5_config.video:
        print("=> generating video, may take some times")
        merge_video(yolo5_config.output)
        
    print("=> main task finished: {}".format(datetime.now().strftime('%H:%M:%S')))
    
    
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['empty','detect','track','dense','count'], default='detect')
    
    parser.add_argument('--input', type=str, default="inference/counter.mp4", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="inference/output", help='folder to save result imgs, can not use input folder')
    parser.add_argument('--area', type=str, default=None, help='area restrict path')
    parser.add_argument('--pools',type=int, default=1, help='max pool num')
    parser.add_argument('--video', action='store_true', help='save result to video')

    parser.add_argument('--weights', type=str, default='pytorch_yolov5/weights/yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    
    yolo5_config = parser.parse_args()
    print(yolo5_config.classes)
    main(yolo5_config)

