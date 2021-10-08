"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import streamlit as st
import io
import time
import os
import pandas as pd
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
from argparse import Namespace
from threading import Timer
import time
import glob
import streamlit as st
from PIL import Image
from PIL import ImageFont
from google.cloud import storage
import gcsfs
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import subprocess
import datetime

import requests
import sys
import time
from pathlib import Path
import pandas_gbq
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
import streamlit as st
from google.cloud import storage
import gcsfs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd 
import plotly.express as px
from detecto import core, utils, visualize
from detecto.utils import reverse_normalize, normalize_transform, _is_iterable
from torchvision import transforms
import torchvision
import extra_streamlit_components as stx
from PIL import Image, ImageEnhance
import PIL
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from streamlit_drawable_canvas import st_canvas


gcs = storage.Client()


############## Detecto ###########
##################################
##################################

def header1(url): 
    st.markdown(f'<p style="color:#1261A0;font-size:30px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)

def header2(url): 
    st.markdown(f'<p style="color:#1261A0;font-size:50px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)
    
def header3(url): 
    st.markdown(f'<p style="color:#5E5A80;font-size:25px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)
    
    
    
    
def load_xml(blobs):
    for blob in blobs:
        try:
            blob.download_to_filename("./train_dataset/{}".format((blob.name).split("/")[-1]))
            print(blob.name, " downloaded")
        except:
            print(blob.name, " download failed")
            
def load_images_detecto(blobs):
    for blob in blobs:
        try:
            blob.download_to_filename("./train_dataset/{}".format((blob.name).split("/")[-1]))
            print(blob.name, " downloaded")
        except:
            print(blob.name, " download failed")
    
    
def load_detecto(train_label):
    sql = """
    SELECT bucket_name, image_path , annotation_path, annotation_name, image_name
    FROM `wawa.master-train-dataset`
    WHERE training_label = "{}"
    ;
    """.format(train_label)

    data_frame = pandas_gbq.read_gbq(
        sql,
        project_id="wawa-smart-store")
    
    storage_client = storage.Client()
    
    image_blobs = []
    annotation_blobs = []
    
    for i in data_frame.iterrows():
        image_blobs.append(storage_client.bucket(i[1]["bucket_name"]).blob(i[1]["image_path"]+i[1]["image_name"]))
        annotation_name = ".".join(i[1]["image_name"].split(".")[:-1])+".txt"
        annotation_blobs.append(storage_client.bucket(i[1]["bucket_name"]).blob(i[1]["annotation_path"]+annotation_name))

    load_images_detecto(image_blobs)
    load_xml(annotation_blobs)    
    
    
    
def download_data_detecto():
    
    sql = """
    SELECT training_label
    FROM `wawa.master-train-dataset`
    ;
    """

    data_frame = pandas_gbq.read_gbq(
        sql,
        project_id="wawa-smart-store")
    tl = ["-"]
    tl.extend(list(set(data_frame["training_label"])))
    training_label = st.selectbox(
     'Select training label for dataset',
     tl)
    
    if training_label == "-":
        return
    load_detecto(training_label)

    
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

    
def file_selector():
    storage_client = storage.Client()
    bucket_name='sizzli_warmer_data'
    bucket = storage_client.get_bucket(bucket_name)
    prefix='test/'
    iterator = bucket.list_blobs(delimiter='/', prefix=prefix)
    response = iterator._get_next_page_response()
    data=[]
    for i in response['items']:
        z='gs://'+bucket_name+'/'+i['name']
        data.append(z)
    data=data[1:]
    return data  

def train():
    
    download_data_detecto()
    
    model_name = st.text_input("Enter the model name")
    
    if st.button("Train model") and len(model_name) != 0:
        label=['sizzli_box']
        dataset = core.Dataset('training_dataset/')
        model = core.Model(label)
        
        st.info('Started training the model!')
        
        model.fit(dataset)
        
        st.info('Saving the model!')
        model.save('/home/jupyter/megha/sizzli_detecto/training_streamlit_app/models/{}.pth'.format(model_name))
        
        st.success("Model saved succesfully!")
        st.balloons()

def annotate():
    
    st.title("Training detecto model for sizzli boxes")
    
    uploaded_file = st.file_uploader("Upload Image Files",type=['jpg','png','jpeg'])
        
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        path = './training_dataset/{name}'.format(name=uploaded_file.name)
        image.save(path)
        
        filename = uploaded_file.name
        
        image = utils.read_image('./training_dataset/{name}'.format(name = filename.split('/')[-1]))
        
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:", ("rect", "freedraw", "line","circle", "transform")
        )
        realtime_update = st.sidebar.checkbox("Update in realtime", True)
        # Create a canvas component
        temp_image = Image.open('./training_dataset/{name}'.format(name = filename.split('/')[-1]))
        width, height = temp_image.size
        newsize = (500, 500)
        temp_image = temp_image.resize(newsize)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open('./training_dataset/{name}'.format(name = filename.split('/')[-1])),
            update_streamlit=realtime_update,
            height=500,
            width = 500,
            drawing_mode=drawing_mode,
            key="canvas",
        )

        if canvas_result.json_data is not None:
            jsondata = canvas_result.json_data

            if st.button("Save"):
                
                st.info("Saved Files")
                
                image = Image.open(uploaded_file)
                image.save("./training_dataset/{}".format(filename.split('/')[-1]))
                
                st.write("{}".format(filename.split('/')[-1]))
                
                data = ET.Element('annotation')

                data.set('verified',"yes")
                # Adding a subtag named `Opening`
                # inside our root tag
                element1 = ET.SubElement(data, 'folder')
                element1.text = "Annotation"

                element2 = ET.SubElement(data, 'filename')
                element2.text = str(filename.split('/')[-1])


                element3 = ET.SubElement(data, 'source')
                subelement3 = ET.SubElement(element3, 'database')
                subelement3.text = "Unknown"

                element4 = ET.SubElement(data, 'size')

                subelement4_1 = ET.SubElement(element4, 'width')
                subelement4_1.text = "12"

                subelement4_2 = ET.SubElement(element4, 'height')
                subelement4_2.text = "12"

                subelement4_3 = ET.SubElement(element4, 'depth')
                subelement4_3.text = "12"

                element5 = ET.SubElement(data, 'segmented')
                element5.text = "0"
                
                for i in range(len(jsondata["objects"])):

                    left = jsondata["objects"][i]["left"]
                    top = jsondata["objects"][i]["top"]
                    right = left + jsondata["objects"][i]["width"]
                    bottom = top + jsondata["objects"][i]["height"]
                    
                    element6 = ET.SubElement(data, 'object')

                    subelement6_1 = ET.SubElement(element6, 'name')
                    subelement6_1.text = "sizzli_box"

                    subelement6_2 = ET.SubElement(element6, 'pose')
                    subelement6_2.text = "Unspecified"

                    subelement6_3 = ET.SubElement(element6, 'truncated')
                    subelement6_3.text = "0"

                    subelement6_4 = ET.SubElement(element6, 'difficult')
                    subelement6_4.text = "0"


                    subelement6_5 = ET.SubElement(element6, 'bndbox')

                    subsubelement6_5_1 = ET.SubElement(subelement6_5, 'xmin')
                    subsubelement6_5_1.text = str(left)

                    subsubelement6_5_2 = ET.SubElement(subelement6_5, 'ymin')
                    subsubelement6_5_2.text = str(top)

                    subsubelement6_5_3 = ET.SubElement(subelement6_5, 'xmax')
                    subsubelement6_5_3.text = str(right)

                    subsubelement6_5_4 = ET.SubElement(subelement6_5, 'ymax')
                    subsubelement6_5_4.text = str(bottom)
                    
                b_xml = ET.tostring(data)
                
                st.write("{}.xml".format(filename.split('/')[-1].split('.')[0]))
                
                with open("./training_dataset/{}.xml".format(filename.split('/')[-1].split('.')[0]), "wb") as f:
                    f.write(b_xml)





#########################
#########################
#####Detection Function
#########################
#########################


@torch.no_grad()
def detect(weights='the_best_for_object.pt',  # model.pt path(s)
        source='test.png',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.15,  # NMS IOU threshold
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
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half= False,  # use FP16 half-precision inference
        label = "Filled" 
        ):
    
    
    
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    co_ordinates = []
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img,
                     augment=augment,
                     visualize=increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

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
                        one_co_ordinate = [i.item() for i in xyxy] 
                        one_co_ordinate.append(label)
                        co_ordinates.append(one_co_ordinate)
                        plot_one_box(xyxy, im0, label="", color=colors(c, True), line_thickness=1)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite("./runs/frame.jpg", im0)
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

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    return co_ordinates,im0

##########
#### Image Selector
##########




def image_selector():
    storage_client = storage.Client()
    bucket_name = 'wawa-cabinet-rpi-images'
    bucket = storage_client.get_bucket(bucket_name)
    prefix='images/'
    iterator = bucket.list_blobs(delimiter='/', prefix=prefix)
    response = iterator._get_next_page_response()
    data=[]
    for i in response['items']:
        z='gs://'+bucket_name+'/'+i['name']
        data.append(z)
    data=data[1:]
    return data 




##########
#### Upload File to GCP
##########


def upload_file_to_gcp(path):
    g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
    temporary_location = path
            
    with open(temporary_location, 'wb') as vid:  ## Open temporary file as bytes
        vid.write(g.read())  ## Read bytes into file
    vid.close()
    gcs.get_bucket('can_detection_data').blob('videos/{name1}'.format(name1= uploaded_file.name)).upload_from_filename('./upload_video/{name}'.format(name=uploaded_file.name))
    
    
##########
#### collecting part out
##########

def draw_boxes(co_ordinates, image):
    for co in co_ordinates:
        
        if co[-1] == "Empty":
            plot_one_box(co[:4], image, label="", color=[0, 0, 250], line_thickness=2)
        else:
            plot_one_box(co[:4], image, label="", color=[0, 255, 0], line_thickness=2)
    
    return image

def outside(one_co_ordinate,exclude):
    cen_x = (one_co_ordinate[0]+one_co_ordinate[2])/2
    cen_y = (one_co_ordinate[1]+one_co_ordinate[3])/2
    return exclude[0] > cen_x or exclude[1] > cen_y or exclude[2] < cen_x or exclude[3] < cen_y



def row_and_exclude(co_ordinates, exclude, get2door = True):
    new_co = []
    
    for co in co_ordinates:
        if(outside(co,exclude) and exclude[0] != -1):
            continue
        new_co.append(co)
    
    
    ##### sorting the co-ordinates
    def sort_key(l):
        return l[1]
    
    sorted_co = sorted(new_co, key  = sort_key)
    
    if len(sorted_co):
        avg_height = (sum([(i[3]-i[1]) for i in sorted_co]))/len(sorted_co)
        avg_width = (sum([(i[2]-i[0]) for i in sorted_co]))/len(sorted_co)
        
    ##### RowWise divisioning
    
    row_wise = [[]]
    now_y_min = sorted_co[0][1]
    cen = (sorted_co[0][1]+sorted_co[0][3])/2
    idx = 1
    row = 1
    
    row_wise_real = [[]]
    
    for xyxy in sorted_co:
        idx += 1
        
        if cen < xyxy[1]:
            idx = 1
            cen = (xyxy[1]+xyxy[3])/2
            now_y_min = xyxy[1]
            row += 1
            row_wise.append([])
            row_wise_real.append([])
        cen = cen*(idx-1)+(xyxy[1]+xyxy[3])/2
        cen = cen/idx
        row_wise[len(row_wise)-1].\
                append([xyxy[0],now_y_min,xyxy[0]+avg_width,now_y_min+avg_height,xyxy[4]])
        row_wise_real[len(row_wise_real)-1].\
                append(xyxy)
    
    ### Row Wise soring 
    print(avg_height, avg_width)
    
    for r in row_wise_real:
        r.sort(key = sort_x)
        
    print(row_wise_real)
        
    door1,door2 = get_2_door(row_wise_real, height = avg_height, width = avg_width)
    
    door1_co = []
    
    door2_co = []
    
    for r in door1:
        for xyxy in r:
            door1_co.append(xyxy)
        
    for r in door2:
        for xyxy in r:
            door2_co.append(xyxy)
    
    image1 = draw_boxes(door1_co, np.array(Image.open("./frame/frame.jpg")))
    
    image2 = draw_boxes(door2_co, np.array(Image.open("./frame/frame.jpg")))
    
    if get2door :
        col1, col2 = st.columns(2)
        col1.header("Left Door")
        col1.image(image1)
        col2.header("Right Door")
        col2.image(image2)
    
    
    return sorted_co, row_wise
    
    
    
    
##########
#### Running inference and showing resultant video
##########

def headergreen(url): 
    st.markdown(f'<p style="color:#00ff00;font-size:48px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)
    
def headerred(url): 
    st.markdown(f'<p style="color:#ff3434;font-size:48px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)


def show(data, resize_frac):
    left = data["objects"][0]["left"]/resize_frac
    top = data["objects"][0]["top"]/resize_frac
    right = left+data["objects"][0]["width"]/resize_frac
    bottom = top+data["objects"][0]["height"]/resize_frac
    video = cv2.VideoCapture("./upload_video/{}".\
                             format(file_path.split("/")[-1]))
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    
    output_file = "./out.mp4"
    
    fps = 20
    
    count = 0
    out_width = int(right-left)
    out_height = int(bottom-top)
    resize_frac = 300/out_width
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V')\
                          , fps, (out_width*2, out_height))
    
    video = cv2.VideoCapture("./upload_video/{}".\
                             format(file_path.split("/")[-1]))
    
    Grid = None
    
    video = cv2.VideoCapture("./upload_video/{}".\
                             format(file_path.split("/")[-1]))
    headergreen("Object")
    headerred("Empty")
    
    while(True):
        succ, frame = video.read()
        
        count += 1
        
        print(count)
        
        if succ :
            image = Image.fromarray(frame)
            image.save("./frame/frame.jpg")
            
            co_ordinates,im0 = detect(source = "./frame/frame.jpg" )
            empty_co_ordinates,im0 = detect(weights = "the_best_empty.pt", source = "./frame/frame.jpg", label = "Empty") 
            
            co_ordinates.extend(empty_co_ordinates)
            new_co,row_wise = row_and_exclude(co_ordinates, [left,top,right,bottom])
            
            image_white = make_white(image, row_wise) 
            
            image = draw_boxes(new_co, np.array(image))
            image = Image.fromarray(image)
            image = image.crop((left, top, right, bottom))
            image = image.resize((out_width,out_height))
            
            image_white = image_white.crop((left, top, right, bottom))
            
            
            image_comb = cv2.hconcat([np.array(image),np.array(image_white)])
            
            image_comb = Image.fromarray(image_comb).resize((out_width*2, out_height))
            
            out.write(np.array(image_comb))
            
        
        else:
            break    
    
    video.release()
    out.release()
    
    output_file = "./out.mp4"
    output_file_streamlit = "./frame/out_st.mp4"
    
    os.system('ffmpeg -y -i {} -vcodec libx264 {}'.format(output_file,output_file_streamlit))
    
    video_file = open('./frame/out_st.mp4','rb')
    video_bytes = video_file.read()
    
    st.video(video_bytes)
    
    
    create_summery(row_wise,get_planogram())
    
    
##########
#### Retraining
########## 


def write_files(empty, filled):
    image = Image.open("./frame/frame.jpg")
    h = image.height()
    w = image.width()
    
    file1 = open("./retrain/image1.txt", "w") 
    
    for ann in empty:
        height = ann[3]-ann

        
        
    
def load_labels(blobs):
    t_len = int(0.7*len(blobs))
    for blob in blobs[:t_len]:
        try:
            blob.download_to_filename("./working/data/labels/train/{}".format((blob.name).split("/")[-1]))
            print(blob.name, " downloaded")
        except:
            print(blob.name, " download failed")
    for blob in blobs[t_len:]:
        try:
            blob.download_to_filename("./working/data/labels/valid/{}".format((blob.name).split("/")[-1]))
            print(blob.name, " downloaded")
        except :
            print(blob.name, " download failed")
            
def load_images(blobs):
    t_len = int(0.7*len(blobs))
    for blob in blobs[:t_len]:
        try:
            blob.download_to_filename("./working/data/images/train/{}".format((blob.name).split("/")[-1]))
            print(blob.name, " downloaded")
        except:
            print(blob.name, " download failed")
    for blob in blobs[t_len:]:
        try:
            blob.download_to_filename("./working/data/images/valid/{}".format((blob.name).split("/")[-1]))
            print(blob.name, " downloaded")
        except:
            print(blob.name, " download failed")
    
    
def load_yolo(train_label):
    sql = """
    SELECT bucket_name, image_path , annotation_path, annotation_name, image_name
    FROM `wawa.master-train-dataset`
    WHERE training_label = "{}"
    ;
    """.format(train_label)

    data_frame = pandas_gbq.read_gbq(
        sql,
        project_id="wawa-smart-store")
    
    storage_client = storage.Client()
    
    image_blobs = []
    annotation_blobs = []
    
    for i in data_frame.iterrows():
        image_blobs.append(storage_client.bucket(i[1]["bucket_name"]).blob(i[1]["image_path"]+i[1]["image_name"]))
        annotation_name = ".".join(i[1]["image_name"].split(".")[:-1])+".txt"
        annotation_blobs.append(storage_client.bucket(i[1]["bucket_name"]).blob(i[1]["annotation_path"]+annotation_name))

    load_images(image_blobs)
    load_labels(annotation_blobs)
    
#     label_blobs = storage_client.list_blobs("wawa-cabinet-retrain", prefix = "yolo/labels/", delimiter='/')
#     label_blobs = [blob for blob in label_blobs]
#     label_blobs = label_blobs[1:]
#     load_labels(label_blobs)
        
    

    
def download_data():
    
    dir_lst = ['./working/data/labels/train','./working/data/labels/valid','./working/data/images/train','./working/data/images/valid']
    for dir in dir_lst:    
        for f in os.listdir(dir):
            try:
                os.remove(os.path.join(dir, f))

            except:
                print("unable to delete ")
    
    sql = """
    SELECT training_label
    FROM `wawa.master-train-dataset`
    ;
    """

    data_frame = pandas_gbq.read_gbq(
        sql,
        project_id="wawa-smart-store")
    tl = ["-"]
    tl.extend(list(set(data_frame["training_label"])))
    training_label = st.selectbox(
     'Select training label for dataset',
     tl)
    
    if training_label == "-":
        return
    load_yolo(training_label)


    
    
def load_weight():
    sql = """
    SELECT training_label
    FROM `wawa.master-weight-dataset`
    ;
    """

    data_frame = pandas_gbq.read_gbq(
        sql,
        project_id="wawa-smart-store")
    tl = ["-"]
    tl.extend(list(set(data_frame["training_label"])))
    training_label = st.selectbox(
     'Select training label for weight',
     tl)
    
    if training_label == "-":
        return
    sql = """
    SELECT bucket_name, weight_path, weight_name
    FROM `wawa.master-weight-dataset`
    WHERE training_label = "{}"
    ;
    """.format(training_label)

    data_frame_weight = pandas_gbq.read_gbq(
        sql,
        project_id="wawa-smart-store")
    
    wl = ["-"]
    wl.extend(list(set(data_frame_weight["weight_name"])))
    weight_label = st.selectbox(
     'Select weight',
     wl)
    
    wt_file = st.file_uploader("Or Upload Weight file")
    wt_local_name = None
    if wt_file :
        bytes_data = wt_file.read()
        st.write("filename:", wt_file.name)
        g = io.BytesIO(wt_file.read())  ## BytesIO Object
        wt_local_name = "./retrain/weights/weight_upload.pt"
        with open(wt_local_name, 'wb') as vid:  ## Open temporary file as bytes
            vid.write(g.read())  ## Read bytes into file
        vid.close()
    if weight_label != "-":
        sql = """
        SELECT bucket_name, weight_path, weight_name
        FROM `wawa.master-weight-dataset`
        WHERE training_label = "{}"
        AND weight_name = "{}"
        ;
        """.format(training_label, weight_label)

        data_frame_weight = pandas_gbq.read_gbq(
            sql,
            project_id="wawa-smart-store")
        
        data_frame_weight = data_frame_weight.values.tolist()[0]
        storage_client = storage.Client()
        wt_local_name = "./retrain/weights/weight_upload.pt"
        storage_client.bucket(data_frame_weight[0]).blob(data_frame_weight[1]+"/"+data_frame_weight[2]).\
        download_to_filename(wt_local_name)
        
        epochs = st.selectbox(
            'Select the number of Epochs',
            ["50","100","300","500"]
        )
        
    if wt_local_name :
        os.system("python train.py --weights={} --epochs={}".format(wt_local_name,epochs))
        try :
            storage_client.bucket("wawa-cabinet-retrain").blob("yolo/weights").upload_from_filename("./runs/train/peet/weights/best.pt")
        except : 
            print("error occured")
            
        time_now = str(pd.Timestamp.now())
        
        df = pd.DataFrame(
                {
                    "date_time": [time_now],
                    "weight_name": [training_label+time_now+".pt"],
                    "weight_path": ["yolo/wieghts"],
                    "bucket_name":["wawa-cabinet-retrain"],
                    "training_label": [training_label]
                }
            )

        df.to_gbq("wawa.master-weight-dataset","wawa-smart-store", if_exists="append")
    
def retrain_util(json_data):
    
    Object_type = st.sidebar.radio(
        "annotation Type :", ("Object", "Empty")
    )
    
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("rect", "Replace and Delete")
    )
    
    if drawing_mode == "Replace and Delete" :
        drawing_mode = "transform"
        st.write("Note: Double Click to Delete a anotation")
    
    if Object_type == "Object":
        stroke_color_detect = "#0cf22b"
    else:
        stroke_color_detect = "#f22b0c"
        
        
    bg_image = Image.open("./frame/frame.jpg")
    canvas_result2 = st_canvas(
        fill_color="rgba(255, 165, 0, 0.15)",  # Fixed fill color with some opacity
        stroke_width=1,
        stroke_color=stroke_color_detect,
        background_color="#eee",
        background_image=bg_image ,
        update_streamlit=True,
        height=bg_image.height,
        width=bg_image.width,
        drawing_mode= drawing_mode,
        initial_drawing = json_data,
        key="annote",
    )

    object_annote = []
    empty_annote = []
    if canvas_result2.json_data is not None:
        data = canvas_result2.json_data
        if len(data["objects"]) != 0:
            for i in range(len(data["objects"])):
                left = data["objects"][i]["left"]
                top = data["objects"][i]["top"]
                width = data["objects"][i]["width"]
                height = data["objects"][i]["height"]
                right = left+data["objects"][i]["width"]
                bottom = top+data["objects"][i]["height"]
                x_center = (left+right)/2;
                y_center = ()
                w = bg_image.width
                h = bg_image.height
                if data["objects"][i]["stroke"] == "#0cf22b" :
                    object_annote.append([left/w,top/h,width/w,height/h])
                else :
                    empty_annote.append([left,top,right,bottom])

    
#     print(len(empty_annote))
#     print(object_annote)
    
    if st.checkbox("Save and Retrain", False) :
        st.write("")
        download_data()
        load_weight()
        
        
        #write_files(empty_annote, object_annote)



def get_canvas_init(co):
    new_json = {"objects" : []}
        
    true = True
    false = False
        
    obj_proto = {
            "type":"rect",
            "version":"4.4.0",
            "originX":"left",
            "originY":"top",
            "left":0,
            "top":0,
            "width":0,
            "height":0,
            "fill":"rgba(255, 165, 0, 0.15)",
            "stroke":"#0cf22b",
            "strokeWidth":1,
            "strokeDashArray":None,
            "strokeLineCap":"butt",
            "strokeDashOffset":0,
            "strokeLineJoin":"miter",
            "strokeUniform":true,
            "strokeMiterLimit":4,
            "scaleX":1,
            "scaleY":1,
            "angle":0,
            "flipX":false,
            "flipY":false,
            "opacity":1,
            "shadow":None,
            "visible":true,
            "backgroundColor":"",
            "fillRule":"nonzero",
            "paintFirst":"fill",
            "globalCompositeOperation":"source-over",
            "skewX":0,
            "skewY":0,
            "rx":0,
            "ry":0,
    }
        
    for i in co:
        dic = obj_proto.copy()
            
        dic["left"] = i[0]
        dic["top"] = i[1]
        dic["width"] = i[2]-i[0]
        dic["height"] = i[3]-i[1]
            
        if i[4] == "Filled":
            dic["stroke"] = "#0cf22b"
            
        else:
            dic["stroke"] = "#f22b0c"
            
        new_json["objects"].append(dic)
        
    return new_json


def retrain():
    co_ordinates,im0 = detect(source = "./frame/frame.jpg" )
    empty_co_ordinates,im0 = detect(weights = "the_best_empty.pt", source = "./frame/frame.jpg", label = "Empty") 
    
    co_ordinates.extend(empty_co_ordinates)
    
    json_data = get_canvas_init(co_ordinates)
    
    retrain_util(json_data)
    
    
    
    
    

##########
#### Image for WebCam
##########    
    
    
    
    
import os, glob
import shutil
def delete():
    with os.scandir("./runs/detect/") as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
    with os.scandir("./runs/train/") as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
    
    with os.scandir("./training_dataset/") as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
    
    
    
##### Main Function #######
############################
if __name__ == "__main__":
    ##### this for branding
    
    genre = st.radio("Select the model to train",
                    ("Detecto","Yolo"))
    
    
    if genre == "Yolo":
        col1, col2, col3 = st.columns([1,4,1])
        with col1:
            st.write("")
        with col2:
            st.image("/home/jupyter/megha/sizzli_detecto/streamlit-app/wawa-logo.png")
        with col3:
            st.write("")
        st.sidebar.write("**Powered By**")
        st.sidebar.image("/home/jupyter/megha/sizzli_detecto/streamlit-app/Techolution-logo.png")

        val = stx.stepper_bar(steps=["Annotate", "generate new model"])
        
        delete()
        uploaded_file = st.file_uploader("Background image:", type=["png", "jpg"])

        ########################



        ### Uploading video 

        if uploaded_file:
            image = Image.open(uploaded_file)
            image = Image.fromarray(np.array(image))
            image.save("./frame/frame.jpg")
        ####################

        ########################

        Conf_Thres_object =  st.sidebar.slider("Object Detection Threshold", 0.3,0.7,0.4)
        Conf_Thres_empty =  st.sidebar.slider("Empty Detection Threshold", 0.3,0.7,0.4)

        if uploaded_file:
            retrain()
    if genre == "Detecto" :
        col1, col2, col3 = st.columns([1,4,1])
        with col1:
            st.write("")
        with col2:
            st.image("/home/jupyter/megha/sizzli_detecto/streamlit-app/wawa-logo.png")
        with col3:
            st.write("")
        st.sidebar.write("**Powered By**")
        st.sidebar.image("/home/jupyter/megha/sizzli_detecto/streamlit-app/Techolution-logo.png")

        val = stx.stepper_bar(steps=["Annotate", "generate new model"])

        if val== 0:
            annotate()

        if val==1:
            train()
        
        
        
        
        
        
        
        
        
        
