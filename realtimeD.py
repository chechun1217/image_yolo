import argparse
import time
from pathlib import Path
import pandas as pd
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import copy
import sys
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

def mayPOS1(fin, realtime):
    lenR = len(realtime)
    lenF = len(fin)
    Fin = copy.deepcopy(fin)
    #Fin = fin
    same_class = 0
    if lenR == 0 or lenF == 0:
        return False
    for i in range (lenR):
        if realtime[i] in Fin :
            same_class = same_class+1
            Fin.remove(realtime[i])
    per = same_class/lenR #算兩個位置相似度------->改成以長度大的當分母  在fin種類較多 or realtime種類較多時 都適用
    if per > 0.6:
        return True#如果相似度>60 %
    else:
        return False

def RSSIcircle(fin_label, RSSI_label):
    col = int(fin_label[:-3])
    row = int(fin_label[-3:-1])
    RSSI_col = int(RSSI_label[:-2])
    RSSI_row = int(RSSI_label[-2:])
    if(abs(RSSI_col - col) + abs(RSSI_row - row) > 3):
        return False
    else:
        return True


def match_item(fin_obj, realtime_obj):
    if len(fin_obj ) == 0 or len(realtime_obj) == 0 or(fin_obj[0] != realtime_obj[0]):
        return 999
    elif (fin_obj[0] == realtime_obj[0]):
        #看class是不是一樣
        return(eucliDist(fin_obj[1], realtime_obj[1]))#對座標做 similarity 讓class名一樣且靠近的obj可以match
 





def detect(save_img=False):
    
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_txt = 1
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        #view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    x = 0
    class_use = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    realtime = pd.DataFrame(columns = ['Label',  'class_all','obj0', 'obj1', 'obj2', 'obj3', 'obj4', 'obj5', 'obj6', 'obj7', 'obj8', 'obj9','obj10', 'obj11', 'obj12', 'obj13', 'obj14', 'obj15', 'obj16', 'obj17', 'obj18', 'obj19','obj20', 'obj21', 'obj22', 'obj23', 'obj24', 'obj25', 'obj26', 'obj27', 'obj28', 'obj29'])
    #csv_data = pd.read_csv('fingerprint.csv',dtype=object)
    per_30_frame=[]
    now_sec=1#看是不是第一秒
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        realtime.loc[x]=''
        #realtime.loc[x, 'Label']=Path(path).stem
        realtime.loc[x, 'class_all']=[]
        
        count = 0
        
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image 
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            

            p = Path(p)  # to Path

            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    #print("s:",s,'c:',c,'n',n,'det',det,'-det',det[:,-1])
                #print(s)
                # Write results
                
                
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        if(names[int(cls)] in class_use): 

                            obj_info = [names[int(cls)], xywh[0:2], xywh[2:]]#每一個東西進來都是一個obj_info  [name,cord,wh]
                            realtime[f'obj{count}'] = realtime[f'obj{count}'].astype('object')
                            realtime.loc[x, f'obj{count}'] = obj_info


                            count = count+1
                            realtime.loc[x, 'class_all'].append(names[int(cls)])
                            
                    
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                
                #在這裡預測位置
                with open('../walk/rssi_result.txt','r') as file:
                    RSSILABEL = file.read()
                    RSSILABEL=RSSILABEL[0:-1]
                with open('../yolov7/data.pkl','rb') as file:
                   fin_df=pickle.load(file)
                RT_class = realtime.loc[x, 'class_all']
                #print(RT_class)
                df_maybe = fin_df[fin_df['class_all'].apply(mayPOS1,args=(RT_class,))]
                #print('1111111111',df_maybe)
                df_maybe = df_maybe[df_maybe['Label'].apply(RSSIcircle,args=(RSSILABEL,))]
                #print('2222222222',df_maybe)
                df_maybe.reset_index(inplace=True)

                predict_score=[]
                RT_len=len(RT_class)
                
                
                for i in range(len(df_maybe)):
                    score_per_row = 0
                    df_len = len(df_maybe.loc[i, 'class_all'])

                    for j in range (RT_len):
                        real_obj = realtime.loc[x, f'obj{j}']
                        match_score = []

                        for m in range(df_len):
                            fin_obj = df_maybe.loc[i, f'obj{m}']
                            score = match_item(fin_obj, real_obj)#------------------------------------------------------------>如果real [a,a,b] fin[a,b]  而real的第二個a實際是對應到fin的a   這樣會出錯 因為real 的第1個a在迴圈先跑 會直接和fin a 配對
                            match_score.append(score)#在這一個位置的所有obj中 找到最像的一個obj 計算cos similarity
                        
                        if min(match_score) == 999:
                            pos_score = 0
                        else:
                            H = np.argmin(match_score)#和finerprint裡第H個obj最像
                            f = eucliDist(df_maybe.loc[i, f'obj{H}'][1], real_obj[1])
                            g = eucliDist(df_maybe.loc[i, f'obj{H}'][2], real_obj[2])
                            
                            
                            if f == 0 or g == 0:
                                pos_score = 1000000000 #大小 位置 完全一樣的話  通常只有在測試的時候會發生
                            else:
                                pos_score = (1/f) * (1/g)#距離越小 分數越高
                                
                            df_maybe.loc[i, f'obj{H}'][0] = 'matched'#已經配對到的 不會再用來配對
                        score_per_row = score_per_row+pos_score    
                    predict_score.append(score_per_row)   
                        
                #print(predict_score)
                if predict_score != []:
                    predict_label = df_maybe.loc[np.argmax(predict_score), 'Label']
                    #print('max score:',max(predict_score))
                    #print('pos:',predict_label)
                    per_30_frame.append(predict_label)


                
                if len(per_30_frame) == 30:
                    #--------------------------------------------particle filter
                    '''
                    dict={}
                    for key in per_30_frame:
                        dict[key]=dict.get(key,0)+1
                    import os
                    import image_Particle_Filter as PF
                    final_position=PF.calculate(dict,count)
                    print('FFF:',final_position)
                    '''
                    label = max(per_30_frame,key=per_30_frame.count)
                    print("Label:",label)
                    sys.stdout.flush()
                    #---------------------------------------------
                    '''
                    label = max(per_30_frame,key=per_30_frame.count)
                    col = int(label[:-3])
                    row = int(label[-3:-1])
                    dir = label[-1]

                    if now_sec == 1:
                        precol = col
                        prerow = row
                        predir = dir
                        print(label)
                        now_sec = now_sec+1
                    elif(abs(precol - col) + abs(prerow - row) > 3):
                        #---------拿RSSI最新預測到的位置來看
                        rssi_pos = '1105'
                        rssi_col = int(rssi_pos[:-2])
                        rssi_row = int(rssi_pos[-2:])
                        pre_diff = abs(precol - rssi_col) + abs(prerow - rssi_row)
                        now_diff = abs(col - rssi_col) + abs(row - rssi_row)
                        #--------------------------------
                        

                        #如果要上一秒
                        if(pre_diff < now_diff):
                            if prerow < 10:
                                label = str(precol) + '0' + str(prerow) + predir
                            else:
                                label = str(precol) + str(prerow) + predir
                            print(label)
                            now_sec = now_sec + 1
                        #如果要這一秒
                        else:
                            precol = col
                            prerow = row
                            predir = dir
                            print(label)
                            now_sec = now_sec + 1
                    else:
                        precol = col
                        prerow = row
                        predir = dir
                        print(label)
                        now_sec = now_sec + 1
                        
                    '''
                    #print('label:',label)
                    per_30_frame = []
                    #如果 上一秒跟這秒位置差太多  就用RSSI程式來看要維持上秒or用這秒

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            '''
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            '''
            # Save results (image with detections)
            '''
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            '''

        x=x+1
    '''
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    '''
    #print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')#------------------------------->改0.8看看
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results',default=False)#改false看看
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    #print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
