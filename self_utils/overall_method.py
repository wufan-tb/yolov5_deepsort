import cv2,os,natsort
import numpy as np

class Object_Counter:
    def __init__(self,name_list) -> None:
        super().__init__()
        self.name_list=name_list
        self.last_id={"P":[],
                      "N":[]}
        self.counter_data={"P":{key:0 for key in name_list},
                           "N":{key:0 for key in name_list}}
        
    def counter_update(self,name,ID,velocity,positive_direction,negetive_direction):
        cosin=np.dot(velocity,positive_direction)/(np.sqrt(np.dot(positive_direction,positive_direction))*np.sqrt(np.dot(velocity,velocity)))
        if cosin >= 0.6 and name in self.name_list and ID not in self.last_id["P"]:
            self.counter_data["P"][name]+=1
            self.last_id["P"].append(ID)
        else:
            cosin=np.dot(velocity,negetive_direction)/(np.sqrt(np.dot(negetive_direction,negetive_direction))*np.sqrt(np.dot(velocity,velocity)))
            if cosin >= 0.6 and name in self.name_list and ID not in self.last_id["N"]:
                self.counter_data["N"][name]+=1
                self.last_id["N"].append(ID)
                
    def draw_counter(self,img,color=[100,250,100],thickness=None,fontsize=None):
        thickness = max(2,round(0.0016 * (img.shape[0] + img.shape[1]) / 2)) if thickness==None else thickness
        fontsize = 0.36*thickness if fontsize==None else fontsize
        top=(5,5)
        text_info="     "
        placeholder="                 "
        for name in self.name_list:
            text_info+=" {} ".format(name)
        t_size=cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , thickness+2)[0]
        cv2.putText(img, text_info, (top[0], top[1]+t_size[1]+2), cv2.FONT_HERSHEY_TRIPLEX, fontsize, color, thickness)
        top=(top[0],top[1]+t_size[1]+10)
        for direction in self.counter_data.keys():
            text_info="{}    ".format(direction)
            for name in self.name_list:
                length=int(len(name)/2)+1
                text_info+=placeholder[0:length]+"{}".format(self.counter_data[direction][name])+placeholder[0:length]
            t_size=cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , thickness+2)[0]
            cv2.putText(img, text_info, (top[0], top[1]+t_size[1]+2), cv2.FONT_HERSHEY_TRIPLEX, fontsize, color, thickness)
            top=(top[0],top[1]+t_size[1]+10)
        return img
                    
class Count_Line:
    def __init__(self,start_point,end_point) -> None:
        super().__init__()
        self.start_point=start_point
        self.end_point=end_point
        unit_vector=np.array(self.end_point)-np.array(self.start_point)
        unit_vector=unit_vector/np.sqrt(np.dot(unit_vector,unit_vector))
        self.positive_direction=np.array([0-unit_vector[1],unit_vector[0]])
        self.negetive_direction=np.array([unit_vector[1],0-unit_vector[0]])
        
    def box_on_line(self,box):
        right_shift=0.5
        down_shift=0.5
        center=[int((1-down_shift)*box[1]+down_shift*box[3]),int((1-right_shift)*box[0]+right_shift*box[2])]
        n=np.array(center)-np.array(self.start_point)
        v=np.array(self.end_point)-np.array(center)
        consin=np.dot(n,v)/(np.sqrt(np.dot(n,n))*np.sqrt(np.dot(v,v)))
        return consin >= 0.999
    
    def draw_line(self,img,onTarget):
        thickness=max(4,round(0.004 * (img.shape[0] + img.shape[1]) / 2))
        if onTarget:
            cv2.line(img,(self.start_point[1],self.start_point[0]),(self.end_point[1],self.end_point[0]), (0,200,0),thickness+2, 8)
        else:
            cv2.line(img,(self.start_point[1],self.start_point[0]),(self.end_point[1],self.end_point[0]), (0,0,200),thickness, 8)
        return img
        
class Area_Restrict:
    def __init__(self,area_restrict,origin_shape) -> None:
        super().__init__()
        self.area_restrict=(area_restrict is not None) and (os.path.isfile(area_restrict))
        if self.area_restrict:
            self.area_path=area_restrict
            area_img = cv2.imread(self.area_path)
            same_size_mask = cv2.resize(area_img, (origin_shape[1], origin_shape[0]))
            _, mask = cv2.threshold(cv2.cvtColor(same_size_mask, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
            self.mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            self.size_trans=area_img.shape[0]/origin_shape[0]
            
            area_img = cv2.cvtColor(area_img, cv2.COLOR_BGR2GRAY)
            _, area_img = cv2.threshold(area_img, 150, 255, cv2.THRESH_BINARY)
            self.area_coords_list = np.argwhere(area_img > 250).tolist()
            
    def box_in_area(self,box):
        if self.area_restrict:
            right_shift=0.5
            down_shift=0.8
            center=[int(self.size_trans*((1-down_shift)*box[1]+down_shift*box[3])),int(self.size_trans*((1-right_shift)*box[0]+right_shift*box[2]))]
            return center in self.area_coords_list
        else:
            return True
    
    def draw_bounding(self,img):
        if self.area_restrict:
            temp = cv2.drawContours(img, self.mask_contours, -1, (10, 10, 250), max(3,round(0.003 * (img.shape[0] + img.shape[1]) / 2)))
            return temp
        else:
            return img
        
class Image_Capture:
    def __init__(self,source) -> None:
        super().__init__()
        self.source=os.path.dirname(source) if (source.endswith(".jpg") or source.endswith(".png")) else source
        if os.path.isdir(self.source):
            self.source_type="imgs" 
        elif source.isdigit():
            self.source_type="camera"
            self.source=int(self.source)
        elif source.startswith("rtsp") or source.startswith("rtmp"):
            self.source_type="camera"
        else:
            self.source_type="video"
        self.index=0
        self.ret=True

        if self.source_type == "imgs":
            if (source.endswith(".jpg") or source.endswith(".png")):
                self.img_List=[os.path.basename(source)]
            else:
                self.img_List=natsort.natsorted(os.listdir(source))
            _,img,_=self.read()
            self.index-=1
            self.shape=img.shape
        else:
            self.cap=cv2.VideoCapture(self.source)
        
    def read(self):
        if self.source_type == "imgs":
            img=cv2.imread(os.path.join(self.source,self.img_List[self.index]))
            ret = True if hasattr(img, 'shape') else False
            self.index+=1
            self.ret=ret
            return ret,img,self.img_List[self.index-1]
        elif self.source_type == "camera":
            ret,img=self.cap.read()
            self.index+=1
            self.ret=ret
            return ret,img,"frame_{}.jpg".format(self.index)
        else:
            ret,img=self.cap.read()
            self.ret=ret
            return ret,img,"frame_{}.jpg".format(int(self.cap.get(1)))
            
    def get(self,i=0):
        if self.source_type == "imgs":
            if i==1:
                return self.index
            if i==7:
                return len(self.img_List)
            if i==4:
                return self.shape[0]
            if i==3:
                return self.shape[1]
            
        elif self.source_type == "camera":
            return self.index if i==1 else int(self.cap.get(i))
        
        else:
            return int(self.cap.get(i))
    
    def get_index(self):
        return self.get(1)
    
    def get_length(self):
        return self.get(7)
    
    def get_height(self):
        return self.get(4)
    
    def get_width(self):
        return self.get(3)
    
    def ifcontinue(self):
        if self.source_type == "imgs":
            return (self.index < len(self.img_List)) and self.ret
        else:
            return (self.cap.get(1) < self.cap.get(7) or self.cap.get(7) <= 0) and self.ret

    def release(self):
        if self.source_type == "imgs":
            pass
        else:
            self.cap.release()
                
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