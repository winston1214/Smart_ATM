import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import warnings
import tkinter #UI
import tkinter.font #UI
import threading #UI
import imutils #UI
warnings.filterwarnings('ignore')


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from facial_recognition.facial_model2 import Facial_model2
from facial_recognition.facial_model3 import Facial_model3
from tkinter import * #UI
from PIL import ImageTk, Image #UI


account_past = pd.read_csv('./my_data/account1.csv')
account_today = pd.read_csv('./my_data/account2.csv')
loan_past = pd.read_csv('./my_data/loan1.csv')
loan_today = pd.read_csv('./my_data/loan2.csv')
insurance_past = pd.read_csv('./my_data/insurance1.csv')
insurance_today = pd.read_csv('./my_data/insurance2.csv')
card_past = pd.read_csv('./my_data/card1.csv')
card_today = pd.read_csv('./my_data/card2.csv')

inout_score = [] #UI
calling_score = [] #UI
facial_score = [] #UI
mydata_score = [] #UI
danger_scores = [] #UI
cap = cv2.VideoCapture #UI
counters = 0 #UI
govideo = True #UI
callingswitch = False #UI
facialswitch = False #UI
mydataswitch = False #UI
calling_count = 0 #UI
danger_count = 0 #UI

def outlier_iqr(data):
    q1, q2, q3 = np.percentile(data,[25,50,75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    
    return q2, q3, upper_bound


def account_data():
    account_past_id = account_past[(account_past['id']==id) & (account_past['type']==1) & (account_past['class']==2)]
    account_today_id = account_today[(account_today['id']==id) & (account_today['type']==1) & (account_today['class']==2)]
    account = 1
    #여수신계좌정보 점수
    if len(account_today_id) == 0:
        account = 0
    else:
        if any(account_today_id['value'] > outlier_iqr(account_past_id['value'])[2]):
            account = 30
    if account == 1:
        if any(outlier_iqr(account_past_id['value'])[1] < account_today_id['value']):
            if any(account_today_id['value'] < outlier_iqr(account_past_id['value'])[2]):
                account = 20
        if any(outlier_iqr(account_past_id['value'])[0] < account_today_id['value']):
            if any(account_today_id['value'] < outlier_iqr(account_past_id['value'])[1]):
                account = 10
        else:
            account = 0

    return account

def loan_data():
    # 여수신대출정보
    loan_past_id = loan_past[loan_past['id']==id]
    loan_today_id = loan_today[loan_today['id']==id]
    loan = 1

    # 여수신대출정보 점수
    if len(loan_today_id) == 0:
        loan = 0
    else:
        if any(loan_today_id['value'] > outlier_iqr(loan_past_id['value'])[2]):
            loan = 20
    if loan == 1:
        if any(outlier_iqr(loan_past_id['value'])[1] < loan_today_id['value']):
            if any(loan_today_id['value'] < outlier_iqr(loan_past_id['value'])[2]):
                loan = 10
        if any(outlier_iqr(loan_past_id['value'])[0] < loan_today_id['value']):
            if any(loan_today_id['value'] < outlier_iqr(loan_past_id['value'])[1]):
                loan = 5
        else:
            loan = 0
    return loan

def insurance_data():
    # 보험대출정보
    insurance_past_id = insurance_past[insurance_past['id']==id]
    insurance_today_id = insurance_today[insurance_today['id']==id]

    insurance = 1
    # 보험대출정보 점수
    if len(insurance_today_id) == 0:
        insurance = 0
    else:
        if any(insurance_today_id['value'] > outlier_iqr(insurance_past_id['value'])[2]):
            insurance = 20
        if insurance == 1:
            if any(outlier_iqr(insurance_past_id['value'])[1] < insurance_today_id['value']):
                if any(insurance_today_id['value'] < outlier_iqr(insurance_past_id['value'])[2]):
                    insurance = 10
            if any(outlier_iqr(insurance_past_id['value'])[0] < insurance_today_id['value']):
                if any(insurance_today_id['value'] < outlier_iqr(insurance_past_id['value'])[1]):
                    insurance = 5
            else:
                insurance = 0
    return insurance

def card_short_loan():
    #카드 대출정보
    card_past_short_id = card_past[(card_past['id']==id) & (card_past['short_loan']== True)]
    card_today_short_id = card_today[(card_today['id']==id) & (card_today['short_loan']== True)]
    card_short = 1

    #카드 단기대출정보 점수
    if len(card_today_short_id) == 0:
        card_short = 0
    else:
        if any(card_today_short_id['short_loan_value'] > outlier_iqr(card_past_short_id['short_loan_value'])[2]):
            card_short = 15
        if card_short == 1:
            if any(outlier_iqr(card_past_short_id['short_loan_value'])[1] < card_today_short_id['short_loan_value']):
                if any(card_today_short_id['short_loan_value'] < outlier_iqr(card_past_short_id['short_loan_value'])[2]):
                    ard_short = 10
            if any(outlier_iqr(card_past_short_id['short_loan_value'])[0] < card_today_short_id['short_loan_value']):
                if any(card_today_short_id['short_loan_value'] < outlier_iqr(card_past_short_id['short_loan_value'])[1]):
                    card_short = 5
            else:
                card_short = 0
    return card_short

def card_long_loan():
    card_past_long_id = card_past[(card_past['id']==id) & (card_past['long_loan']== True)]
    card_today_long_id = card_today[(card_today['id']==id) & (card_today['long_loan']== True)]
    card_long = 1

    #카드 장기대출정보 점수
    if len(card_today_long_id) == 0:
        card_long = 0
    else:
        if any(card_today_long_id['long_loan_value'] > outlier_iqr(card_past_long_id['long_loan_value'])[2]):
            card_long = 15

        if card_long == 1:
            if any(outlier_iqr(card_past_long_id['long_loan_value'])[1] < card_today_long_id['long_loan_value']):
                if any(card_today_long_id['long_loan_value'] < outlier_iqr(card_past_long_id['long_loan_value'])[2]):
                    card_long = 10
            if any(outlier_iqr(card_past_long_id['long_loan_value'])[0] < card_today_long_id['long_loan_value']):
                if any(card_today_long_id['long_loan_value'] < outlier_iqr(card_past_long_id['long_loan_value'])[1]):
                    card_long = 5
            else:
                card_long = 0

    return card_long


def calling(): #UI
    global callingswitch
    global facialswitch

    callingswitch = True
    facialswitch = False
    playvideo()
    button1['state'] = tkinter.DISABLED
    button2['state'] = tkinter.DISABLED
    button3['state'] = tkinter.DISABLED

def facial(): #UI
    global govideo
    global callingswitch
    global facialswitch

    callingswitch = False
    facialswitch = True
    govideo = True
    button2['state'] = tkinter.DISABLED

def mydata(): #UI
    global govideo
    global callingswitch
    global facialswitch
    global mydataswitch
    global danger_count

    callingswitch = False
    facialswitch = False
    mydataswitch = True
    danger_count = 0
    govideo = True
    button3['state'] = tkinter.DISABLED

def watcher(): #UI
    global govideo

    timer = threading.Timer(0.1, watcher)
    timer.start()

    if govideo == True:
        timer.cancel()
        videolabel.after(10, playvideo)

def playvideo(): #UI
    global inout_score
    global calling_score
    global facial_score
    global mydata_score
    global danger_scores
    global counters
    global cap
    global govideo
    global callingswitch
    global facialswitch
    global calling_count
    global danger_count

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    frame = imutils.resize(frame, width=1280)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    videolabel.imgtk = imgtk
    videolabel.configure(image=imgtk)
    if govideo:
        videolabel.after(10, playvideo)
    else:
        watcher()

    if govideo and callingswitch:
        if calling_score[counters] == 30:
            calling_count += 1
            if calling_count >= 60:
                govideo = False
                warning_popup()
                button2['state'] = tkinter.NORMAL

    if govideo and facialswitch:
        if facial_score[counters] == 30:
            danger_count += 1
            if danger_count >= 50:
                govideo = False
                button3['state'] = tkinter.NORMAL

    if govideo and mydataswitch:
        if danger_scores[counters] > 70:
            danger_count += 1
            
            if danger_count >= 30:
                govideo = False
                text5.delete(1.0, END)
                text5.insert(END, danger_scores[counters])
                Danger_popup()

    if counters % 30 == 0:
        if mydataswitch:
            text4.delete(1.0, END)
            text4.insert(END, mydata_score[counters])


        if callingswitch:
            text2.delete(1.0, END)
            text2.insert(END, calling_score[counters])

        if facialswitch:
            text3.delete(1.0, END)
            text3.insert(END, facial_score[counters])

        text1.delete(1.0, END)
        text1.insert(END, inout_score[counters])
        
        window.update()
    counters += 1

def warning_popup(): #UI
   top= Toplevel(window)
   top.title("Warning")
   top.geometry("600x300+500+350")
   top.resizable(False, False)
   top.configure(bg=background_color)

   popup_label1 = tkinter.Label(top, bg=background_color, fg=font_color, font=fontStyle, text='Please take off your mask')
   popup_label1.place(x=120, y=100)
   popup_button2 = tkinter.Button(top, text='OK', overrelief='solid', width=10, font=fontStyle, state=tkinter.NORMAL, command=lambda:[top.destroy()])
   popup_button2.place(x=210, y=200)

def Danger_popup(): #UI
   top= Toplevel(window)
   top.title("Warning")
   top.geometry("600x300+500+350")
   top.resizable(False, False)
   top.configure(bg=background_color)

   popup_label1 = tkinter.Label(top, bg=background_color, fg=font_color, font=fontStyle, text='High risk of voicephishing!!!')
   popup_label1.place(x=120, y=100)
   popup_button2 = tkinter.Button(top, text='OK', overrelief='solid', width=10, font=fontStyle, state=tkinter.NORMAL, command=lambda:[top.destroy()])
   popup_button2.place(x=210, y=200)

@torch.no_grad()
def run(id=101,
        facial_weights_file = ROOT/ 'weights/facial_best.pt',
        weights=ROOT / 'weights/detect_best.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):


    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    # facial recognition
    if 'jm' in source or 'ym' in source:
        #print('ym')
        face_model = Facial_model3().to(device)
        facial_weights_file = 'weights/tmp1.pt'
        face_model.load_state_dict(torch.load(facial_weights_file))
    if 'yk' in source or 'dj' in source:
        face_model = Facial_model2().to(device)
        face_model.load_state_dict(torch.load(facial_weights_file))

    call_check = 0
    call_hand = []
    call_hand_loc = ''
    calling_weights = 0
    calling = False
    face_weights = 0
    face_cls = ['Normal','Danger','Happy']
    mydata_weights = 0
    danger_count = 0
    danger_facial = 0
    pred_emotion = ''

    global inout_score #UI
    global calling_score #UI
    global facial_score #UI
    global mydata_score #UI
    global danger_scores #UI
    global cap  # UI

    account = account_data()  #여수신계좌정보
    loan = loan_data() #여수신대출정보
    insurance = insurance_data() #보험대출정보
    card_short = card_short_loan() #카드단기대출정보
    card_long = card_long_loan() #카드장기대출정보
    mydata_weights = 0.6 * (account + loan + insurance + card_short + card_long)  #금융데이터 점수


    for path, im, im0s, vid_cap, s in dataset:
        face_weights = 0
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            frame_ls = []
            mask_face = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                        if label == 'mask' or 'face':
                            mask_face.append(label)
                        xyxy_ = list(map(lambda x: x.item(),xyxy))
                        # xyxy_.append(conf) # 신뢰도
                        xyxy_.append(names[c]) # label
                        
                        frame_ls.append(xyxy_)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                label_ls = list(map(lambda x: x[-1],frame_ls))
                # if label_ls.count('mask') > 1:
                    


                if ('mask' in label_ls and 'hand' in label_ls): # check call
                    
                    hand_xyxy = list(filter(lambda x: x[-1] == 'hand',frame_ls))
                    hand_xyxy = list(map(lambda x: x[:-1],hand_xyxy))
                    hand_left_x ,hand_right_x = np.array(hand_xyxy,dtype=np.int32)[:,0], np.array(hand_xyxy,dtype=np.int32)[:,2]                
                    mask_xyxy = list(filter(lambda x: x[-1] == 'mask',frame_ls))
                    mask_xyxy = list(map(lambda x: x[:-1],mask_xyxy))
                    mask_left_x ,mask_right_x = np.array(mask_xyxy,dtype=np.int32)[:,0],np.array(mask_xyxy,dtype=np.int32)[:,2]
                    
                    if len(hand_left_x) >1: # 손이 두개 나올 때 왼손 오른손 구분
                        if call_hand.count('left')>call_hand.count('right'):
                            call_hand_loc = 'left' 
                            hand_left_x = sorted(hand_left_x)[0]
                            hand_right_x = sorted(hand_right_x)[0]
                        else:
                            call_hand_loc = 'right'
                            hand_left_x = sorted(hand_left_x)[-1]
                            hand_right_x = sorted(hand_right_x)[-1]

                    # if mask_left_x > hand_left_x: # 영상의 왼쪽에서 전화 받음
                    #     call_hand.append('left')
                    #     if mask_left_x < hand_right_x < mask_right_x:
                    #         call_check += 1
                    #         if call_check >= 100:
                    #             #cv2.putText(im0, 'Calling', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    #             #cv2.putText(im0,'Take Off mask',(100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    #             calling_weights = 30.0

                    #             calling = True
                                
                    
                    if mask_right_x < hand_right_x: # 영상의 오른쪽에서 전화 받음
                        call_hand.append('right')
                        if mask_left_x < hand_left_x < mask_right_x:
                            call_check += 1
                            if call_check >= 1000:
                                #cv2.putText(im0, 'Calling', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                                #cv2.putText(im0,'Take Off mask',(100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                                calling_weights = 30.0

                                calling = True
                        
                            

                elif ('face' in label_ls and 'hand' in label_ls):
                    
                    hand_xyxy = list(filter(lambda x: x[-1] == 'hand',frame_ls))
                    hand_xyxy = list(map(lambda x: x[:-1],hand_xyxy))
                    hand_left_x ,hand_right_x = np.array(hand_xyxy,dtype=np.int32)[:,0], np.array(hand_xyxy,dtype=np.int32)[:,2] 

                    face_xyxy = list(filter(lambda x: x[-1] == 'face',frame_ls))
                    face_xyxy = list(map(lambda x: x[:-1],face_xyxy))
                    face_left_x ,face_right_x = np.array(face_xyxy)[:,0],np.array(face_xyxy)[:,2]

                    face_crop = save_one_box(face_xyxy, im0, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True,save = False)
                    

                    if len(hand_left_x) >1: # 손이 두개 나올 때 왼손 오른손 구분
                        if call_hand.count('left')>call_hand.count('right'):
                            hand_left_x = sorted(hand_left_x)[0]
                            hand_right_x = sorted(hand_right_x)[0]
                            call_hand_loc = 'left'
                        else:
                            call_hand_loc = 'right'
                            hand_left_x = sorted(hand_left_x)[-1]
                            hand_right_x = sorted(hand_right_x)[-1]

                    if face_left_x > hand_left_x : # 영상 왼쪽에서 전화 받음
                        call_hand.append('left')
                        if face_left_x <= hand_right_x <= face_right_x:
                            call_check += 1
                            if call_check >= 30:
                                #cv2.putText(im0, 'Calling', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                                calling_weights = 30.0
                                calling = True
                    if face_right_x < hand_right_x:
                        call_hand.append('right')
                        if face_left_x <= hand_left_x <= face_right_x:
                            call_check += 1
                            if call_check >= 30:
                                #cv2.putText(im0,'Calling', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                                calling_weights = 30.0
                                calling = True
                    if calling:
                        face_crop = cv2.resize(face_crop, dsize=(482,482),interpolation=cv2.INTER_LINEAR)
                        image_swap = np.swapaxes(face_crop, 0,2)
                        image_swap = np.expand_dims(image_swap, axis=0)
                        tensor = torch.from_numpy(image_swap).type(torch.FloatTensor).to(device)
                        face_model.eval()
                        output = F.softmax(face_model(tensor))
                        output = output.cpu().detach().numpy().flatten()

                        pred_emotion = face_cls[np.argmax(output)]
                        
                        
                        if pred_emotion == 'danger':
                            face_weights = output[1]*10 # 표정 가중치


                        else:
                            face_weights = output[1]*10

                        #if danger_facial > 10:
                            # cv2.putText(im0,'danger',(100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        # else:
                            # cv2.putText(im0,f'{pred_emotion}',(100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                         

                if 'hand' not in label_ls:
                    call_check = 0
                    calling = False

                if len(call_hand) >= 100:
                    call_hand.pop()
                
                if calling :
                    calling_weights = 30

                if calling == False:
                    calling_weights = 30

            danger_score = sum([calling_weights,face_weights,mydata_weights]) # 여기에 점수 변수 넣어줘
            inout_score.append(account*0.6) #UI
            calling_score.append(calling_weights) #UI
            facial_score.append(face_weights) #UI
            mydata_score.append(mydata_weights) #UI
            danger_scores.append(danger_score)  #UI

            #print('bank',mydata_weights)
            #print('face',face_weights)
            #print('calling_weights',calling_weights)
            #print(danger_score)
            #print(danger_count)
            #cv2.putText(im0,f'Danger score: %.2f'%danger_score,(250,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if danger_score > 0.7:
                #cv2.putText(im0,'Danger',(100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                danger_count +=1
                #cv2.putText(im0,f'Danger score: %.2f'%danger_score,(250,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            #if danger_count == 10:  #안되서 주석처리함
                #root = tk.Tk()
                #tk.messagebox.showwarning('Voice Pshing',"Danger Account")
                #root.destroy()
            elif danger_count > 10:
                cv2.putText(im0,'Danger',(100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else:
                cv2.putText(im0,f'{pred_emotion}',(100,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    vid_writer[i].release() #UI
    cap = cv2.VideoCapture(save_path) #UI

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=101, help='ID number')
    parser.add_argument('--facial-weights-file', type=str, default='weights/facial_best.pt', help='facial recognition weights file')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/detect_best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    id = opt.id
    main(opt)

    background_color = '#272c39'
    font_color = '#ffffff'

    window = tkinter.Tk()  # UI
    window.title("test")
    window.geometry("1780x880+100+100")
    window.resizable(False, False)
    window.configure(bg=background_color)

    fontStyle = tkinter.font.Font(weight="bold", size=20)
    fontStyle1 = tkinter.font.Font(size=30)

    frame1 = tkinter.Frame(window)
    frame1.grid(row=0, column=0, rowspan=2)
    frame1.configure(bg=background_color)
    frame2 = tkinter.Frame(window)
    frame2.grid(row=0, column=1)
    frame2.configure(bg=background_color)
    frame3 = tkinter.Frame(window)
    frame3.grid(row=1, column=1)
    frame3.configure(bg=background_color)

    photo = PhotoImage(file="./video-background1280x848.png")
    videolabel = tkinter.Label(frame1, image=photo)
    videolabel.grid(row=0, column=0, rowspan=6, padx=5, pady=5)
    label1 = tkinter.Label(frame2, bg=background_color, fg=font_color, font=fontStyle, text='Financial score')
    label1.grid(row=1, column=1, padx=4, pady=30)
    label2 = tkinter.Label(frame2, bg=background_color, fg=font_color, font=fontStyle, text='Calling score')
    label2.grid(row=2, column=1, padx=4, pady=30)
    label3 = tkinter.Label(frame2, bg=background_color, fg=font_color, font=fontStyle, text='Facial score')
    label3.grid(row=3, column=1, padx=4, pady=30)
    label4 = tkinter.Label(frame2, bg=background_color, fg=font_color, font=fontStyle, text='MYDATA score')
    label4.grid(row=4, column=1, padx=4, pady=30)
    label5 = tkinter.Label(frame2, bg=background_color, fg=font_color, font=fontStyle, text='Danger score')
    label5.grid(row=5, column=1, padx=4, pady=30)

    text1 = tkinter.Text(frame2, height=1, width=10, font=fontStyle1)
    text1.grid(row=1, column=2, columnspan=2, padx=5, pady=10)
    text2 = tkinter.Text(frame2, height=1, width=10, font=fontStyle1)
    text2.grid(row=2, column=2, columnspan=2, padx=5, pady=10)
    text3 = tkinter.Text(frame2, height=1, width=10, font=fontStyle1)
    text3.grid(row=3, column=2, columnspan=2, padx=5, pady=10)
    text4 = tkinter.Text(frame2, height=1, width=10, font=fontStyle1)
    text4.grid(row=4, column=2, columnspan=2, padx=5, pady=10)
    text5 = tkinter.Text(frame2, height=1, width=10, font=fontStyle1)
    text5.grid(row=5, column=2, columnspan=2, padx=5, pady=10)

    text1.insert(END, '0.0')
    text2.insert(END, '0.0')
    text3.insert(END, '0.0')
    text4.insert(END, '0.0')
    text5.insert(END, '0.0')

    button1 = tkinter.Button(frame3, text='Calling', overrelief='solid', width=7, font=fontStyle, state=tkinter.NORMAL, command=calling)
    button1.grid(row=5, column=1, padx=5, pady=1)
    button2 = tkinter.Button(frame3, text='Facial', overrelief='solid', width=7, font=fontStyle, state=tkinter.NORMAL, command=facial)
    button2.grid(row=5, column=3, padx=5, pady=1)
    button3 = tkinter.Button(frame3, text='Mydata', overrelief='solid', width=7, font=fontStyle, state=tkinter.NORMAL, command=mydata)
    button3.grid(row=5, column=5, padx=5, pady=1)

    window.mainloop()
