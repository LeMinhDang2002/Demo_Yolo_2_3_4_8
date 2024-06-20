import argparse
import cv2
import os
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw
import asyncio, os
import imutils
import pandas as pd
import torch
from st_pages import add_page_title, hide_pages


import torch

# load transform
from dataset.build import build_transform

# load some utils
from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import visualize

from models.detectors import build_model
from config import build_model_config, build_trans_config, build_dataset_config


class_names = ['motobike']
num_classes = len(class_names)
keep_prob = 0.7
epsilon = 1e-07

from GFPGAN.gfpgan.utils import GFPGANer
def func_GFPGAN(input_img, bg_upsampler = 'realesrgan', bg_tile = 400, version = '1.3', upscale = 2, weight = 0.5):
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            # import warnings
            # warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
            #               'If you really want to use it, please modify the corresponding codes.')
            # bg_upsampler = None
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=False)  # need to set False in CPU mode
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None


    # ------------------------ set up GFPGAN restorer ------------------------
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')
    

    # determine model paths
    model_path = os.path.join('GFPGAN/experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url


    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        # has_aligned=args.aligned,
        # only_center_face=args.only_center_face,
        paste_back=True,
        weight=weight)
    
    restorer = None
    return restored_img


def are_lines_parallel(angle_deg, threshold = 2):
    if np.abs(angle_deg) < threshold:
        return True
    return False
def are_lines_perpendicular(angle_deg, threshold = 2):
    if np.abs(np.abs(angle_deg) - 90) < threshold:
        return True
    return False

class Args():
    def __init__(self):
        self.img_size = 640
        self.mosaic = None
        self.mixup = None
        self.mode = 'image'
        self.cuda = False
        self.show = False
        self.gif = False
        # Model setting
        self.model = 'yolov8_n'
        self.num_classes = 1
        self.weight = './Weights/yolov8_n_last_mosaic_epoch.pth'
        self.conf_thresh = 0.35
        self.nms_thresh = 0.5
        self.topk = 100
        self.deploy = False
        self.fuse_conv_bn = False
        self.no_multi_labels = False
        self.nms_class_agnostic = False
        # Data Setting
        self.dataset = 'plate_number'

def Rerun(final_image, cnn, threshold = 170):
    img_gray_lp = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

    LP_WIDTH = final_image.shape[1]
    LP_HEIGHT = final_image.shape[0]

    #estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/14,
                        LP_WIDTH/4,
                        LP_HEIGHT/3,
                        LP_HEIGHT/2]

    # _, img_binary_lp = cv2.threshold(img_gray_lp, 140, 255, cv2.THRESH_BINARY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, threshold, 255, cv2.THRESH_BINARY)

    cntrs, _ = cv2.findContours(img_binary_lp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Approx dimensions of the contours
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    #Check largest 15 contours for license plate character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]


    character = []
    x_cntr_list_1 = []
    x_cntr_list_2 = []
    target_contours = []
    img_res_1 = []
    img_res_2 = []

    rotate_locations = []

    for cntr in cntrs :
        #detecting contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        intX-=5
        intY-=5
        intWidth = int(intWidth*1.2)
        intHeight = int(intHeight*1.1)
        
        #checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height and intY < LP_HEIGHT/3  and intX > 0 and intY > 0:
            x_cntr_list_1.append(intX) 
            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = final_image[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (75, 100))
            cv2.rectangle(img_binary_lp, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 1)
            img_res_1.append(char) # List that stores the character's binary image (unsorted)

    for cntr in cntrs :
        #detecting contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        intX-=5
        intY-=5
        intWidth = int(intWidth*1.2)
        intHeight = int(intHeight*1.1)
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height and intY > LP_HEIGHT/3 and intX>0 and intY > 0:
            # print(intX, intY, intWidth, intHeight)
            rotate_locations.append([intX, intY])
            x_cntr_list_2.append(intX) 
            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = final_image[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (75, 100))
            cv2.rectangle(img_binary_lp, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 1)
            img_res_2.append(char) # List that stores the character's binary image (unsorted)

    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list_1)), key=lambda k: x_cntr_list_1[k])
    # indices = sorted(range(len(x_cntr_list)), key=lambda k: (y_cntr_list[k], x_cntr_list[k]))
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res_1[idx])# stores character images according to their index
    img_res_1 = np.array(img_res_copy)

    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list_2)), key=lambda k: x_cntr_list_2[k])
    # indices = sorted(range(len(x_cntr_list)), key=lambda k: (y_cntr_list[k], x_cntr_list[k]))
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res_2[idx])# stores character images according to their index
    img_res_2 = np.array(img_res_copy)

    if(len(img_res_1) != 0 and len(img_res_2) != 0):
        img_res = np.concatenate((img_res_1, img_res_2), axis=0)
    elif (len(img_res_1) != 0 and len(img_res_2) == 0):
        img_res = img_res_1
    elif (len(img_res_1) == 0 and len(img_res_2) != 0):
        img_res = img_res_2
    for i in range(len(img_res)):

        # Chuyển đổi độ sâu của hình ảnh sang định dạng 8-bit unsigned integer
        normalized_image = cv2.normalize(img_res[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        resized_finalimage = cv2.resize(normalized_image, (75, 100))

        resized_finalimage = np.expand_dims(resized_finalimage, axis=0)
        predicts = cnn.predict(resized_finalimage)
        predicted_class = np.argmax(predicts, axis=1)
        print(predicted_class[0])

        if (predicted_class[0]) >= 10:
            character.append(chr((predicted_class[0] - 10) + ord('A')))
        else:
            character.append(predicted_class[0])

    char_array = [str(item) for item in character]
    result_string = ''.join(char_array[:])

    return img_binary_lp, result_string

def DisplayDemo(yolo, cnn, uploaded_files):
    args = Args()
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])
    data_cfg  = build_dataset_config(args)

    ## Data info
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    class_indexs = data_cfg['class_indexs']
    val_transform, trans_cfg = build_transform(args, trans_cfg, model_cfg['max_stride'], is_train=False)

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.write("Filename:", uploaded_file.name)
        st.write("Image shape:", img.shape)
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        image_real = img.copy()
        
        orig_h, orig_w, _ = image_real.shape
        img_transform, _, ratio = val_transform(image_real)
        img_transform = img_transform.unsqueeze(0).to(device)

        # inference
        outputs = yolo(img_transform)
        scores = outputs['scores']
        labels = outputs['labels']
        bboxes = outputs['bboxes']

        bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)
        x = int((int(bboxes[0][0]) + int(bboxes[0][2]))/2)
        y = int((int(bboxes[0][1]) + int(bboxes[0][3]))/2) 
        w = int(bboxes[0][2] - bboxes[0][0]) * 1.2
        h = int(bboxes[0][3] - bboxes[0][1]) * 1.1

        # x_min = int(bboxes[0][0])
        # y_min = int(bboxes[0][1])
        # x_max = int(bboxes[0][2])
        # y_max = int(bboxes[0][3])
        x_min, y_min = int(x - w / 2), int(y - h / 2)
        x_max, y_max = int(x + w / 2), int(y + h / 2)

        img_draw = Image.fromarray(img)
        draw = ImageDraw.Draw(img_draw)

        draw.rectangle([x_min, y_min, x_max, y_max], outline='red')

        # Hiển thị hình ảnh với hình vẽ
        st.image(img_draw, caption='Hình ảnh với hình vẽ', use_column_width=True)

        cropped_image = img[y_min:y_max, x_min:x_max]
        cropped_image = cv2.resize(cropped_image, (115, 100), interpolation = cv2.INTER_AREA)

        restore_img = func_GFPGAN(input_img=cropped_image, upscale=6)

        image_copy = restore_img.copy()
        
        # Convert image to grayscale
        gray = cv2.cvtColor(restore_img,cv2.COLOR_BGR2GRAY)
        # Use canny edge detection
        edges = cv2.Canny(gray,100,200,apertureSize=3)
        lines = cv2.HoughLinesP(
                    edges, # Input edge image
                    1, # Distance resolution in pixels
                    # np.pi/180, # Angle resolution in radians
                    np.pi/120, # Angle resolution in radians
                    threshold=120, # Min number of votes for valid line
                    minLineLength=250, # Min allowed length of line
                    maxLineGap= 200 # Max allowed gap between line for joining them
                    )

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg_check = np.degrees(angle_rad)
            # cv2.line(restore_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if y2 >= 0 and y2 < int(restore_img.shape[0]/2) and y1 >= 0 and y1 < int(restore_img.shape[0]/2) and np.abs(angle_deg_check) < 45:
                cv2.line(restore_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
            if y2 > int(restore_img.shape[0]/2) and y2 < int(restore_img.shape[0]) and y1 >  int(restore_img.shape[0]/2) and y1 < int(restore_img.shape[0]) and np.abs(angle_deg_check) < 45:
                cv2.line(restore_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)

        rotated_image = imutils.rotate(image_copy, angle_deg)

        gray = cv2.cvtColor(rotated_image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,200,apertureSize=3)
        lines = cv2.HoughLinesP(
                            edges, # Input edge image
                            1, # Distance resolution in pixels
                            # np.pi/180, # Angle resolution in radians
                            np.pi/120, # Angle resolution in radians
                            # threshold=100, # Min number of votes for valid line
                            threshold=100, # Min number of votes for valid line
                            minLineLength=200, # Min allowed length of line
                            maxLineGap= 300 # Max allowed gap between line for joining them
                            )
        distance_top = 600
        distance_bottom = 600
        y_min, y_max = 0, rotated_image.shape[0]
        deg = 0
        # height_tmp = int(image.shape[1])
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 125), 2)
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            if y2 < int(rotated_image.shape[0]/2) and y1 < int(rotated_image.shape[0]/2) and are_lines_parallel(angle_deg, threshold=3) and y_min < y2:
                # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                y_min = y2
            if y2 > int(rotated_image.shape[0]/2) and y1 > int(rotated_image.shape[0]/2) and are_lines_parallel(angle_deg, threshold=4) and y_max > y2 and y2 > rotated_image.shape[0]/2 + 50:
                # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                y_max = y2

            # if x1 < int(rotated_image.shape[1]/2) and x2 < int(rotated_image.shape[1]/2):
            # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            if are_lines_perpendicular(angle_deg) and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                distance_bottom = 0
                distance_top = 0
                deg = 0
            elif are_lines_perpendicular(angle_deg, threshold=2) == False and np.abs(angle_deg) > 45 and np.abs(angle_deg) > deg:
                # np.tan(np.deg2rad(np.abs(np.abs(angle_deg) - 90)))
                # print(rotated_image.shape[1])
                if x1 <= x2 and y1 > y2 and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                    tmp = int(np.tan(np.deg2rad(np.abs(np.abs(angle_deg) - 90))) * np.abs(y1 - y2)) 
                    if tmp <= distance_top :
                        distance_top = tmp
                        deg = np.abs(angle_deg)
                elif x1 <= x2 and y1 < y2 and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                    tmp = int(np.tan(np.deg2rad(np.abs(np.abs(angle_deg) - 90))) * np.abs(y1 - y2)) 
                    if tmp <= distance_bottom:
                        # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # print(x1, y1, x2, y2)
                        distance_bottom = tmp
                        deg = np.abs(angle_deg)

        if(distance_top != 600):
            # print(y_min, y_max)
            cropped_image = rotated_image[np.abs(y_min - 15):y_max + 15, :]
            # src = rotated_image.copy()
            src = cropped_image.copy()
            srcTri = np.array( [[0, 0], [src.shape[1], 0], [0, src.shape[0]]] ).astype(np.float32)
            # dstTri = np.array( [[0, src.shape[1]]*0, [src.shape[1]-1, src.shape[0]*0], [src.shape[1]*0, src.shape[0]*0.7]] ).astype(np.float32)
            dstTri = np.array( [[-distance_top, 0], [src.shape[1], 0 ], [0 , src.shape[0]]] ).astype(np.float32)
            warp_mat = cv2.getAffineTransform(srcTri, dstTri)
            warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

        if(distance_bottom != 600):
            # print(y_min, y_max)
            cropped_image = rotated_image[np.abs(y_min - 15):y_max + 15, :]
            # src = rotated_image.copy()
            src = cropped_image.copy()
            srcTri = np.array( [[0, 0], [src.shape[1], 0], [0, src.shape[0]]] ).astype(np.float32)
            # dstTri = np.array( [[0, src.shape[1]]*0, [src.shape[1]-1, src.shape[0]*0], [src.shape[1]*0, src.shape[0]*0.7]] ).astype(np.float32)
            dstTri = np.array( [[0, 0], [src.shape[1], 0 ], [-distance_bottom , src.shape[0]]] ).astype(np.float32)
            warp_mat = cv2.getAffineTransform(srcTri, dstTri)
            warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))


        if (distance_top == 600 and distance_bottom == 600):
            gray = cv2.cvtColor(rotated_image,cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,50,200,apertureSize=3)
            lines = cv2.HoughLinesP(
                                edges, # Input edge image
                                1, # Distance resolution in pixels
                                # np.pi/180, # Angle resolution in radians
                                np.pi/120, # Angle resolution in radians
                                # threshold=100, # Min number of votes for valid line
                                threshold=120, # Min number of votes for valid line
                                minLineLength=250, # Min allowed length of line
                                maxLineGap= 300 # Max allowed gap between line for joining them
                                )
            distance_top = 0
            distance_bottom = 0
            x_min, y_min, x_max, y_max = 0, 0, 0,0
            deg = 0
            # height_tmp = int(image.shape[1])
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 125), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)


                if are_lines_parallel(angle_deg) and y1 < int(rotated_image.shape[0]/2):
                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    y_min = y1
                if are_lines_parallel(angle_deg) and y1 > int(rotated_image.shape[0]/2):
                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    y_max = y1
                if are_lines_perpendicular(angle_deg) and x1 < int(rotated_image.shape[1]/2):
                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 125), 2)
                    x_min = x1
                if are_lines_perpendicular(angle_deg) and x1 > int(rotated_image.shape[1]/2):
                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (125, 255, 0), 2)
                    x_max = x1
            cropped_image = rotated_image[y_min:y_max, x_min:x_max]


            img_gray_lp = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            LP_WIDTH = cropped_image.shape[1]
            LP_HEIGHT = cropped_image.shape[0]
            dimensions = [LP_WIDTH/14,
                                LP_WIDTH/4,
                                LP_HEIGHT/3,
                                LP_HEIGHT/2]
            # _, img_binary_lp = cv2.threshold(img_gray_lp, 2, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            _, img_binary_lp = cv2.threshold(img_gray_lp, 170, 255, cv2.THRESH_BINARY)

            cntrs, _ = cv2.findContours(img_binary_lp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #Approx dimensions of the contours
            lower_width = dimensions[0]
            upper_width = dimensions[1]
            lower_height = dimensions[2]
            upper_height = dimensions[3]

            #Check largest 15 contours for license plate character respectively
            cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]


            character = []
            x_cntr_list_1 = []
            x_cntr_list_2 = []
            target_contours = []
            img_res_1 = []
            img_res_2 = []

            rotate_locations = []

            for cntr in cntrs :
                #detecting contour in binary image and returns the coordinates of rectangle enclosing it
                intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
                intX-=5
                intY-=5
                intWidth = int(intWidth*1.2)
                intHeight = int(intHeight*1.1)
                
                #checking the dimensions of the contour to filter out the characters by contour's size
                if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height and intY < LP_HEIGHT/3 :
                    x_cntr_list_1.append(intX) 
                    char_copy = np.zeros((44,24))
                    #extracting each character using the enclosing rectangle's coordinates.
                    char = cropped_image[intY:intY+intHeight, intX:intX+intWidth]
                    char = cv2.resize(char, (75, 100))
                    cv2.rectangle(img_binary_lp, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 1)
                    img_res_1.append(char) # List that stores the character's binary image (unsorted)

            for cntr in cntrs :
                #detecting contour in binary image and returns the coordinates of rectangle enclosing it
                intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
                intX-=5
                intY-=5
                intWidth = int(intWidth*1.2)
                intHeight = int(intHeight*1.1)
                if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height and intY > LP_HEIGHT/3 :
                    # print(intX, intY, intWidth, intHeight)
                    rotate_locations.append([intX, intY])
                    x_cntr_list_2.append(intX) 
                    char_copy = np.zeros((44,24))
                    #extracting each character using the enclosing rectangle's coordinates.
                    char = cropped_image[intY:intY+intHeight, intX:intX+intWidth]
                    char = cv2.resize(char, (75, 100))
                    cv2.rectangle(img_binary_lp, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 1)
                    img_res_2.append(char) # List that stores the character's binary image (unsorted)
        
            #arbitrary function that stores sorted list of character indeces
            indices = sorted(range(len(x_cntr_list_1)), key=lambda k: x_cntr_list_1[k])
            # indices = sorted(range(len(x_cntr_list)), key=lambda k: (y_cntr_list[k], x_cntr_list[k]))
            img_res_copy = []
            for idx in indices:
                img_res_copy.append(img_res_1[idx])# stores character images according to their index
            img_res_1 = np.array(img_res_copy)

            #arbitrary function that stores sorted list of character indeces
            indices = sorted(range(len(x_cntr_list_2)), key=lambda k: x_cntr_list_2[k])
            # indices = sorted(range(len(x_cntr_list)), key=lambda k: (y_cntr_list[k], x_cntr_list[k]))
            img_res_copy = []
            for idx in indices:
                img_res_copy.append(img_res_2[idx])# stores character images according to their index
            img_res_2 = np.array(img_res_copy)

            if(len(img_res_1) != 0 and len(img_res_2) != 0):
                img_res = np.concatenate((img_res_1, img_res_2), axis=0)
            elif (len(img_res_1) != 0 and len(img_res_2) == 0):
                img_res = img_res_1
            elif (len(img_res_1) == 0 and len(img_res_2) != 0):
                img_res = img_res_2
            for i in range(len(img_res)):

                # Chuyển đổi độ sâu của hình ảnh sang định dạng 8-bit unsigned integer
                normalized_image = cv2.normalize(img_res[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                resized_finalimage = cv2.resize(normalized_image, (75, 100))

                resized_finalimage = np.expand_dims(resized_finalimage, axis=0)
                predicts = cnn.predict(resized_finalimage)
                predicted_class = np.argmax(predicts, axis=1)

                if (predicted_class[0]) >= 10:
                    character.append(chr((predicted_class[0] - 10) + ord('A')))
                else:
                    character.append(predicted_class[0])

            char_array = [str(item) for item in character]
            result_string = ''.join(char_array[:])
            st.write(result_string)


            dataframe1 = pd.read_excel('./BANG_SO_XE.xlsx')

            # Chuyển đổi DataFrame thành mảng Python
            data_array = dataframe1.values

            # In ra mảng dữ liệu
            for i in range(len(data_array)):
                if data_array[i][1] == result_string:
                    st.write(data_array[i][0])
                    break
        else:
            gray = cv2.cvtColor(warp_dst,cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,50,200,apertureSize=3)
            lines = cv2.HoughLinesP(
                        edges, # Input edge image
                        1, # Distance resolution in pixels
                        # np.pi/180, # Angle resolution in radians
                        np.pi/120, # Angle resolution in radians
                        # threshold=100, # Min number of votes for valid line
                        threshold=100, # Min number of votes for valid line
                        minLineLength=100, # Min allowed length of line
                        maxLineGap= 200 # Max allowed gap between line for joining them
                        )
            x_min, y_min, x_max, y_max = 0,0,warp_dst.shape[1],warp_dst.shape[0]
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(warp_dst, (x1, y1), (x2, y2), (255, 255, 0), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
                if are_lines_perpendicular(angle_deg, threshold=4) and x1 < int(warp_dst.shape[1]/2) and x1 > 10 and x2 > 10:
                    # cv2.line(warp_dst, (x1, y1), (x2, y2), (0, 255, 125), 2)
                    if x_min != 0 and x1 < x_min:
                        x_min = x1
                    elif x_min == 0:
                        x_min = x1
                if are_lines_perpendicular(angle_deg, threshold=4) and x1 > int(warp_dst.shape[1]/2) and x1 < warp_dst.shape[1]-10 and x2 < warp_dst.shape[1]-10:
                    # cv2.line(warp_dst, (x1, y1), (x2, y2), (125, 255, 0), 2)
                    if x_max != warp_dst.shape[1] and x1 > x_max:
                        x_max = x1
                    elif x_max == warp_dst.shape[1]:
                        x_max = x1
            cropped_image = warp_dst[:, x_min:x_max]


            img_gray_lp = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            LP_WIDTH = cropped_image.shape[1]
            LP_HEIGHT = cropped_image.shape[0]

            #estimations of character contours sizes of cropped license plates
            dimensions = [LP_WIDTH/14,
                                LP_WIDTH/4,
                                LP_HEIGHT/3,
                                LP_HEIGHT/2]

            # _, img_binary_lp = cv2.threshold(img_gray_lp, 2, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            _, img_binary_lp = cv2.threshold(img_gray_lp, 170, 255, cv2.THRESH_BINARY)

            cntrs, _ = cv2.findContours(img_binary_lp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(cntrs)
            #Approx dimensions of the contours
            lower_width = dimensions[0]
            upper_width = dimensions[1]
            lower_height = dimensions[2]
            upper_height = dimensions[3]

            #Check largest 15 contours for license plate character respectively
            cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]


            character = []
            x_cntr_list_1 = []
            x_cntr_list_2 = []
            target_contours = []
            img_res_1 = []
            img_res_2 = []

            rotate_locations = []

            for cntr in cntrs :
                #detecting contour in binary image and returns the coordinates of rectangle enclosing it
                intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
                intX-=5
                intY-=5
                intWidth = int(intWidth*1.2)
                intHeight = int(intHeight*1.1)
                
                #checking the dimensions of the contour to filter out the characters by contour's size
                if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height and intY < LP_HEIGHT/3  and intX > 0 and intY > 0:
                    x_cntr_list_1.append(intX) 
                    char_copy = np.zeros((44,24))
                    #extracting each character using the enclosing rectangle's coordinates.
                    char = cropped_image[intY:intY+intHeight, intX:intX+intWidth]
                    char = cv2.resize(char, (75, 100))
                    cv2.rectangle(img_binary_lp, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 1)
                    img_res_1.append(char) # List that stores the character's binary image (unsorted)

            for cntr in cntrs :
                #detecting contour in binary image and returns the coordinates of rectangle enclosing it
                intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
                intX-=5
                intY-=5
                intWidth = int(intWidth*1.2)
                intHeight = int(intHeight*1.1)
                if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height and intY > LP_HEIGHT/3 and intX>0 and intY > 0:
                    # print(intX, intY, intWidth, intHeight)
                    rotate_locations.append([intX, intY])
                    x_cntr_list_2.append(intX) 
                    char_copy = np.zeros((44,24))
                    #extracting each character using the enclosing rectangle's coordinates.
                    char = cropped_image[intY:intY+intHeight, intX:intX+intWidth]
                    char = cv2.resize(char, (75, 100))
                    cv2.rectangle(img_binary_lp, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 1)
                    img_res_2.append(char) # List that stores the character's binary image (unsorted)
        
            #arbitrary function that stores sorted list of character indeces
            indices = sorted(range(len(x_cntr_list_1)), key=lambda k: x_cntr_list_1[k])
            # indices = sorted(range(len(x_cntr_list)), key=lambda k: (y_cntr_list[k], x_cntr_list[k]))
            img_res_copy = []
            for idx in indices:
                img_res_copy.append(img_res_1[idx])# stores character images according to their index
            img_res_1 = np.array(img_res_copy)

            #arbitrary function that stores sorted list of character indeces
            indices = sorted(range(len(x_cntr_list_2)), key=lambda k: x_cntr_list_2[k])
            # indices = sorted(range(len(x_cntr_list)), key=lambda k: (y_cntr_list[k], x_cntr_list[k]))
            img_res_copy = []
            for idx in indices:
                img_res_copy.append(img_res_2[idx])# stores character images according to their index
            img_res_2 = np.array(img_res_copy)
            
            if(len(img_res_1) != 0 and len(img_res_2) != 0):
                img_res = np.concatenate((img_res_1, img_res_2), axis=0)
            elif (len(img_res_1) != 0 and len(img_res_2) == 0):
                img_res = img_res_1
            elif (len(img_res_1) == 0 and len(img_res_2) != 0):
                img_res = img_res_2
            for i in range(len(img_res)):

                # Chuyển đổi độ sâu của hình ảnh sang định dạng 8-bit unsigned integer
                normalized_image = cv2.normalize(img_res[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                resized_finalimage = cv2.resize(normalized_image, (75, 100))

                resized_finalimage = np.expand_dims(resized_finalimage, axis=0)

                predicts = cnn.predict(resized_finalimage)
                predicted_class = np.argmax(predicts, axis=1)
                print(predicted_class[0])

                if (predicted_class[0]) >= 10:
                    character.append(chr((predicted_class[0] - 10) + ord('A')))
                else:
                    character.append(predicted_class[0])

            char_array = [str(item) for item in character]
            result_string = ''.join(char_array[:])
            if len(result_string) == 0:
                s = f"<p style='font-size:100px; text-align: center'>🥺</p>"
                st.markdown(s, unsafe_allow_html=True) 
            elif len(result_string) > 0 and len(result_string) < 9:
                try:
                    img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 150)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 190)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 130)
                            if(len(result_string) <8):
                                s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả chữ số</p>"
                                st.markdown(s, unsafe_allow_html=True) 
                            else:
                                s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                                st.markdown(s, unsafe_allow_html=True) 
                                df = pd.read_excel('./BANG_SO_XE.xlsx')
                                data_array = df.values
                                for i in range(len(data_array)):
                                    if np.char.strip(data_array[i][1]) == result_string[:4]:
                                        s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                        st.markdown(s, unsafe_allow_html=True)
                                        break
                        else:
                            s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                            st.markdown(s, unsafe_allow_html=True) 
                            df = pd.read_excel('./BANG_SO_XE.xlsx')
                            data_array = df.values
                            for i in range(len(data_array)):
                                if np.char.strip(data_array[i][1]) == result_string[:4]:
                                    s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                    st.markdown(s, unsafe_allow_html=True)
                                    break
                    else:
                        s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                        st.markdown(s, unsafe_allow_html=True) 
                        df = pd.read_excel('./BANG_SO_XE.xlsx')
                        data_array = df.values
                        for i in range(len(data_array)):
                            if np.char.strip(data_array[i][1]) == result_string[:4]:
                                s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                st.markdown(s, unsafe_allow_html=True)
                                break
                except:
                    img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 190)
                    if(len(result_string) <8):
                        s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả chữ số</p>"
                        st.markdown(s, unsafe_allow_html=True) 
                    else:
                        s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                        st.markdown(s, unsafe_allow_html=True) 

                        df = pd.read_excel('./BANG_SO_XE.xlsx')
                        data_array = df.values
                        for i in range(len(data_array)):
                            if np.char.strip(data_array[i][1]) == result_string[:4]:
                                s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                st.markdown(s, unsafe_allow_html=True)
                                break

            elif len(result_string) == 9:
                s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                st.markdown(s, unsafe_allow_html=True) 

                df = pd.read_excel('./BANG_SO_XE.xlsx')
                data_array = df.values
                for i in range(len(data_array)):
                    if np.char.strip(data_array[i][1]) == result_string[:4]:
                        s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                        st.markdown(s, unsafe_allow_html=True)
                        break
            else:
                try:
                    img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 150)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 190)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 130)
                            if(len(result_string) <8):
                                s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả chữ số</p>"
                                st.markdown(s, unsafe_allow_html=True) 
                            else:
                                s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                                st.markdown(s, unsafe_allow_html=True) 
                        else:
                            s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                            st.markdown(s, unsafe_allow_html=True) 

                            df = pd.read_excel('./BANG_SO_XE.xlsx')
                            data_array = df.values
                            for i in range(len(data_array)):
                                if np.char.strip(data_array[i][1]) == result_string[:4]:
                                    s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                    st.markdown(s, unsafe_allow_html=True)
                                    break
                    else:
                        s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                        st.markdown(s, unsafe_allow_html=True) 

                        df = pd.read_excel('./BANG_SO_XE.xlsx')
                        data_array = df.values
                        for i in range(len(data_array)):
                            if np.char.strip(data_array[i][1]) == result_string[:4]:
                                s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                st.markdown(s, unsafe_allow_html=True)
                                break
                except:
                    img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 190)
                    if(len(result_string) <8):
                        s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả chữ số</p>"
                        st.markdown(s, unsafe_allow_html=True) 
                    else:
                        s = f"<p style='font-size:40px;'>🥳 {result_string[:2]}-{result_string[2:4]} {result_string[4:7]}.{result_string[7:]}</p>"
                        st.markdown(s, unsafe_allow_html=True) 

                        df = pd.read_excel('./BANG_SO_XE.xlsx')
                        data_array = df.values
                        for i in range(len(data_array)):
                            if np.char.strip(data_array[i][1]) == result_string[:4]:
                                s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                st.markdown(s, unsafe_allow_html=True)
                                break



add_page_title()
uploaded_files = st.file_uploader("Choose an image file", accept_multiple_files=True)
try:
    DisplayDemo(st.session_state.yolov8, st.session_state.cnn, uploaded_files)
except:
    s = f"<p style='font-size:40px;'>Ảnh không thể nhận diện được</p>"
    st.markdown(s, unsafe_allow_html=True)