import streamlit as st
import cv2, numpy as np, time, math
from utils.kmeans import kmeans, iou_dist, euclidean_dist
from PIL import Image, ImageDraw
import asyncio, os
import imutils
import pandas as pd
import torch
from st_pages import add_page_title, hide_pages
# load transform
from dataset.build import build_transform

from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import visualize

from models.detectors import build_model
from config import build_model_config, build_trans_config, build_dataset_config
add_page_title()

class_names = ['motobike']
num_classes = len(class_names)
keep_prob = 0.7
epsilon = 1e-07

def cal_iou_nms(xywh_true, xywh_pred):
    """Calculate IOU of two tensors.

    Args:
        xywh_true: A tensor or array-like of shape (..., 4).
            (x, y) should be normalized by image size.
        xywh_pred: A tensor or array-like of shape (..., 4).
    Returns:
        An iou_scores array.
    """
    xy_true = xywh_true[..., 0:2] # N*1*1*1*(S*S)*2
    wh_true = xywh_true[..., 2:4]

    xy_pred = xywh_pred[..., 0:2] # N*S*S*B*1*2
    wh_pred = xywh_pred[..., 2:4]
    
    half_xy_true = wh_true / 2.
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = np.maximum(mins_pred,  mins_true)
    intersect_maxes = np.minimum(maxes_pred, maxes_true)
    intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = wh_true[..., 0] * wh_true[..., 1]
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/(union_areas + epsilon)
    
    return iou_scores

# def iou(xywh_true, xywh_pred):

#     x_true = xywh_true[0]
#     y_true = xywh_true[1]
#     w_true = xywh_true[2]
#     h_true = xywh_true[3]

#     x_pred = xywh_pred[0]
#     y_pred = xywh_pred[1]
#     w_pred = xywh_pred[2]
#     h_pred = xywh_pred[3]

#     half_x_true = w_true/2.
#     half_y_true = h_true/2.

#     min_x_true = x_true - half_x_true
#     min_y_true = y_true - half_y_true
#     max_x_true = x_true + half_x_true
#     max_y_true = y_true + half_y_true

#     half_x_pred = w_pred/2.
#     half_y_pred = h_pred/2.

#     min_x_pred = x_pred - half_x_pred
#     min_y_pred = y_pred - half_y_pred
#     max_x_pred = x_pred + half_x_pred
#     max_y_pred = y_pred + half_y_pred

#     intersect_x_min = max(int(min_x_true), int(min_x_pred))
#     intersect_y_min = max(int(min_y_true), int(min_y_pred))

#     intersect_x_max = min(int(max_x_true), int(max_x_pred))
#     intersect_y_max = min(int(max_y_true), int(max_y_pred))

#     intersect_areas = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
#     true_areas = w_true * h_true
#     pred_areas = w_pred * h_pred

#     union_areas = pred_areas + true_areas - intersect_areas
#     iou_scores  = intersect_areas/(union_areas + epsilon)

#     return iou_scores
def iou(xywh_true, xywh_pred, epsilon=1e-6):
    x_true = xywh_true[0]
    y_true = xywh_true[1]
    w_true = xywh_true[2]
    h_true = xywh_true[3]

    x_pred = xywh_pred[0]
    y_pred = xywh_pred[1]
    w_pred = xywh_pred[2]
    h_pred = xywh_pred[3]

    half_w_true = w_true / 2.
    half_h_true = h_true / 2.

    min_x_true = x_true - half_w_true
    min_y_true = y_true - half_h_true
    max_x_true = x_true + half_w_true
    max_y_true = y_true + half_h_true

    half_w_pred = w_pred / 2.
    half_h_pred = h_pred / 2.

    min_x_pred = x_pred - half_w_pred
    min_y_pred = y_pred - half_h_pred
    max_x_pred = x_pred + half_w_pred
    max_y_pred = y_pred + half_h_pred

    intersect_x_min = max(min_x_true, min_x_pred)
    intersect_y_min = max(min_y_true, min_y_pred)
    intersect_x_max = min(max_x_true, max_x_pred)
    intersect_y_max = min(max_y_true, max_y_pred)


    intersect_width = max(0, intersect_x_max - intersect_x_min)
    intersect_height = max(0, intersect_y_max - intersect_y_min)
    intersect_area = intersect_width * intersect_height

    true_area = w_true * h_true
    pred_area = w_pred * h_pred

    union_area = true_area + pred_area - intersect_area
    iou_score = intersect_area / (union_area + epsilon)

    return iou_score


def giou(xywh_true, xywh_pred):
    x_true = xywh_true[0]
    y_true = xywh_true[1]
    w_true = xywh_true[2]
    h_true = xywh_true[3]

    x_pred = xywh_pred[0]
    y_pred = xywh_pred[1]
    w_pred = xywh_pred[2]
    h_pred = xywh_pred[3]

    half_x_true = w_true/2.
    half_y_true = h_true/2.

    min_x_true = x_true - half_x_true
    min_y_true = y_true - half_y_true
    max_x_true = x_true + half_x_true
    max_y_true = y_true + half_y_true

    half_x_pred = w_pred/2.
    half_y_pred = h_pred/2.

    min_x_pred = x_pred - half_x_pred
    min_y_pred = y_pred - half_y_pred
    max_x_pred = x_pred + half_x_pred
    max_y_pred = y_pred + half_y_pred

    intersect_x_min = max(int(min_x_true), int(min_x_pred))
    intersect_y_min = max(int(min_y_true), int(min_y_pred))

    intersect_x_max = min(int(max_x_true), int(max_x_pred))
    intersect_y_max = min(int(max_y_true), int(max_y_pred))

    intersect_areas = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
    true_areas = w_true * h_true
    pred_areas = w_pred * h_pred

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/(union_areas + epsilon)

    C_x_min = min(int(min_x_true), int(min_x_pred))
    C_y_min = min(int(min_y_true), int(min_y_pred))

    C_x_max = max(int(max_x_true), int(max_x_pred))
    C_y_max = max(int(max_y_true), int(max_y_pred))

    C = (C_x_max - C_x_min)*(C_y_max - C_y_min)

    score_giou = iou_scores - ((C - union_areas)/C)

    return score_giou

def diou(xywh_true, xywh_pred):
    x_true = xywh_true[0]
    y_true = xywh_true[1]
    w_true = xywh_true[2]
    h_true = xywh_true[3]

    x_pred = xywh_pred[0]
    y_pred = xywh_pred[1]
    w_pred = xywh_pred[2]
    h_pred = xywh_pred[3]

    half_x_true = w_true/2.
    half_y_true = h_true/2.

    min_x_true = x_true - half_x_true
    min_y_true = y_true - half_y_true
    max_x_true = x_true + half_x_true
    max_y_true = y_true + half_y_true

    half_x_pred = w_pred/2.
    half_y_pred = h_pred/2.

    min_x_pred = x_pred - half_x_pred
    min_y_pred = y_pred - half_y_pred
    max_x_pred = x_pred + half_x_pred
    max_y_pred = y_pred + half_y_pred

    x_heart_true = xywh_true[0]
    y_heart_true = xywh_true[1]
    x_heart_pred = xywh_pred[0]
    y_heart_pred = xywh_pred[1]

    C_x_min = min(int(min_x_true), int(min_x_pred))
    C_y_min = min(int(min_y_true), int(min_y_pred))

    C_x_max = max(int(max_x_true), int(max_x_pred))
    C_y_max = max(int(max_y_true), int(max_y_pred))

    c_square = np.square(C_x_max - C_x_min) + np.square(C_y_max - C_y_min)
    d_square = np.square(x_heart_pred - x_heart_true) + np.square(y_heart_pred - y_heart_true)

    diou = d_square/c_square

    return diou

def cal_iou_v2(xywh_true, xywh_pred):
    """Calculate IOU of two tensors.

    Args:
        xywh_true: A tensor or array-like of shape (..., 4).
            (x, y) should be normalized by image size.
        xywh_pred: A tensor or array-like of shape (..., 4).
    Returns:
        An iou_scores array.
    """
    xy_true = xywh_true[..., 0:2] # N*1*1*1*(S*S)*2
    wh_true = xywh_true[..., 2:4]

    xy_pred = xywh_pred[..., 0:2] # N*S*S*B*1*2
    wh_pred = xywh_pred[..., 2:4]
    
    half_xy_true = wh_true / 2.
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = np.maximum(mins_pred,  mins_true)
    intersect_maxes = np.minimum(maxes_pred, maxes_true)
    intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = wh_true[..., 0] * wh_true[..., 1]
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/(union_areas + epsilon)
    
    return iou_scores



def nms(xywhcp, class_num=1, nms_threshold=0.5):
    """Non-Maximum Suppression.

    Args:
        xywhcp: output from `decode()`.
        class_num:  An integer,
            number of classes.
        nms_threshold: A float, default is 0.5.

    Returns:
        xywhcp through nms.
    """
    argmax_prob = xywhcp[..., 5].astype("int")

    xywhcp_new = []
    for i_class in range(class_num):
        xywhcp_class = xywhcp[argmax_prob==i_class]
        xywhc_class = xywhcp_class[..., :5]
        prob_class = xywhcp_class[..., 6]

        xywhc_axis0 = np.reshape(
            xywhc_class, (-1, 1, 5))
        xywhc_axis1 = np.reshape(
            xywhc_class, (1, -1, 5))

        iou_scores = cal_iou_v2(xywhc_axis0, xywhc_axis1)
        conf = xywhc_class[..., 4]*prob_class
        sort_index = np.argsort(conf)[::-1]

        white_list = []
        delete_list = []
        for conf_index in sort_index:
            white_list.append(conf_index)
            if conf_index not in delete_list:
                iou_score = iou_scores[conf_index]
                overlap_indexes = np.where(iou_score >= nms_threshold)[0]

                for overlap_index in overlap_indexes:
                    if overlap_index not in white_list:
                        delete_list.append(overlap_index)
        xywhcp_class = np.delete(xywhcp_class, delete_list, axis=0)
        xywhcp_new.append(xywhcp_class)
    
    xywhcp_new = sorted(xywhcp_new, reverse=True, key=lambda x:x[0][5])
    #### stage 2: loop over all boxes, remove boxes with high IOU
    xywhcp_final = []
    # while(len(xywhcp_new) > 0):

    ### Này code chữa cháy thôi á chừng nào nhiều hơn 1 đối tượng thì phải code khác
    while(len(xywhcp_new) > 0):
        # print("vô đây ra mà")
        current_box = xywhcp_new.pop(0)
        index = np.argmax(current_box[:, 4])  # Cột thứ 5 (index 4) chứa giá trị cần tìm

        # Lấy mảng có giá trị cao nhất
        max_array = current_box[index]
        xywhcp_final.append(max_array)
        # print(current_box)
        # for box in xywhcp_new:
        #     if( current_box[5] == box[5]):
        #         print(current_box[..., 0:2])
                # print(box[..., 3:4])
                # iou = cal_iou_v2(current_box[:4], box[:4])
                # st.write(iou)
                # if(iou > 0.4):
                #    xywhcp_new_list[0].remove(box)

    xywhcp = np.vstack(xywhcp_final)
    return xywhcp

def soft_nms(xywhcp, class_num=1,
        nms_threshold=0.5, conf_threshold=0.5, sigma=0.5, version = 2):
    """Soft Non-Maximum Suppression.

    Args:
        xywhcp: output from `decode()`.
        class_num:  An integer,
            number of classes.
        nms_threshold: A float, default is 0.5.
        conf_threshold: A float,
            threshold for quantizing output.
        sigma: A float,
            sigma for Soft NMS.

    Returns:
        xywhcp through nms.
    """
    argmax_prob = xywhcp[..., 5].astype("int")

    xywhcp_new = []
    for i_class in range(class_num):
        xywhcp_class = xywhcp[argmax_prob==i_class]
        xywhc_class = xywhcp_class[..., :5]
        prob_class = xywhcp_class[..., 6]

        xywhc_axis0 = np.reshape(
            xywhc_class, (-1, 1, 5))
        xywhc_axis1 = np.reshape(
            xywhc_class, (1, -1, 5))

        iou_scores = cal_iou_nms(xywhc_axis0, xywhc_axis1)
        # conf = xywhc_class[..., 4]*prob_class
        conf = xywhc_class[..., 4]
        sort_index = np.argsort(conf)[::-1]

        white_list = []
        delete_list = []
        for conf_index in sort_index:
            white_list.append(conf_index)
            iou_score = iou_scores[conf_index]
            overlap_indexes = np.where(iou_score >= nms_threshold)[0]

            for overlap_index in overlap_indexes:
                if overlap_index not in white_list:
                    conf_decay = np.exp(-1*(iou_score[overlap_index]**2)/sigma)
                    conf[overlap_index] *= conf_decay
                    if conf[overlap_index] < conf_threshold:
                        delete_list.append(overlap_index)
        xywhcp_class = np.delete(xywhcp_class, delete_list, axis=0)
        xywhcp_new.append(xywhcp_class)
    if version == 3:
        xywhcp_new = sorted(xywhcp_new[0], reverse=True, key=lambda x:x[4])
    if version == 4:
        xywhcp_new = sorted(xywhcp_new[0], reverse=True, key=lambda x:x[3])
    xywhcp = np.vstack(xywhcp_new)
    return xywhcp


def decode(*label_datas,
           class_num=1,
           threshold=0.5,
           version=1):
    """Decode the prediction from yolo model.

    Args:
        *label_datas: Ndarrays,
            shape: (grid_heights, grid_widths, info).
            Multiple label data can be given at once.
        class_num:  An integer,
            number of classes.
        threshold: A float,
            threshold for quantizing output.
        version: An integer,
            specifying the decode method, yolov1、v2 or v3.

    Return:
        Numpy.ndarray with shape: (N, 7).
            7 values represent:
            x, y, w, h, c, class index, class probability.
    """
    output = []
    for label_data in label_datas:
        grid_shape = label_data.shape[:2]
        if version == 1:
            bbox_num = (label_data.shape[-1] - class_num)//5
            xywhc = np.reshape(label_data[..., :-class_num],
                               (*grid_shape, bbox_num, 5))
            prob = np.expand_dims(
                label_data[..., -class_num:], axis=-2)
        elif version == 2 or version == 3:
            bbox_num = label_data.shape[-1]//(5 + class_num)
            label_data = np.reshape(label_data,
                                    (*grid_shape,
                                     bbox_num, 5 + class_num))
            xywhc = label_data[..., :5]
            prob = label_data[..., -class_num:]
        else:
            raise ValueError("Invalid version: %s" % version)   

        # joint_conf = xywhc[..., 4:5]*prob
        joint_conf = xywhc[..., 4:5]
        where = np.where(joint_conf >= threshold)

        for i in range(len(where[0])):
            x_i = where[1][i]
            y_i = where[0][i]
            box_i = where[2][i]
            class_i = where[3][i]

            x_reg = xywhc[y_i, x_i, box_i, 0]
            y_reg = xywhc[y_i, x_i, box_i, 1]
            w_reg = xywhc[y_i, x_i, box_i, 2]
            h_reg = xywhc[y_i, x_i, box_i, 3]
            conf = xywhc[y_i, x_i, box_i, 4]

            x = (x_i + x_reg)/grid_shape[1]
            y = (y_i + y_reg)/grid_shape[0]
            
            w = w_reg
            h = h_reg
            
            if version == 1:
                p = prob[y_i, x_i, 0, class_i]
            else:
                p = prob[y_i, x_i, box_i, class_i]
            output.append([x, y, w, h, conf, class_i, p])
    output = np.array(output, dtype="float")
    return output

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

def Rerun(final_image, cnn, threshold = 170):
    img_gray_lp = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

    LP_WIDTH = final_image.shape[1]
    LP_HEIGHT = final_image.shape[0]

    #estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/14,
                        LP_WIDTH/4,
                        LP_HEIGHT/4,
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

    img_res = []
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
        self.conf_thresh = 0.05
        self.nms_thresh = 0.5
        self.topk = 100
        self.deploy = False
        self.fuse_conv_bn = False
        self.no_multi_labels = False
        self.nms_class_agnostic = False
        # Data Setting
        self.dataset = 'plate_number'
def RunDemoV8(yolo, cnn, image):
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

    img, img_draw, cropped_image = None, None, None
    image_real = image.copy()

    orig_h, orig_w, _ = image_real.shape
    img_transform, _, ratio = val_transform(image_real)
    img_transform = img_transform.unsqueeze(0).to(device)

    # inference
    outputs = yolo(img_transform)
    scores = outputs['scores']
    labels = outputs['labels']
    bboxes = outputs['bboxes']

    index = np.argmax(scores)

    bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)
    x = int((int(bboxes[index][0]) + int(bboxes[index][2]))/2)
    y = int((int(bboxes[index][1]) + int(bboxes[index][3]))/2) 
    w = int(bboxes[index][2] - bboxes[index][0]) * 1.2
    h = int(bboxes[index][3] - bboxes[index][1]) * 1.1

    xywhcc = []
    xywhcc.append(int((int(bboxes[0][0]) + int(bboxes[0][2]))/2))
    xywhcc.append(int((int(bboxes[0][1]) + int(bboxes[0][3]))/2))
    xywhcc.append(int(bboxes[0][2] - bboxes[0][0]))
    xywhcc.append(int(bboxes[0][3] - bboxes[0][1]))
    xywhcc.append(scores[0])
    xywhcc.append(labels[0])

    # x_min = int(bboxes[0][0])
    # y_min = int(bboxes[0][1])
    # x_max = int(bboxes[0][2])
    # y_max = int(bboxes[0][3])
    x_min, y_min = int(x - w / 2), int(y - h / 2)
    x_max, y_max = int(x + w / 2), int(y + h / 2)

    if x_min < 0:
        x_min = 1
    if y_min < 0:
        y_min = 1

    cropped_image = image[y_min:y_max, x_min:x_max]
    if cropped_image.shape[0] <= 115:
        cropped_image = cv2.resize(cropped_image, (115, 100), interpolation = cv2.INTER_AREA)
        # st.image(cropped_image, caption="Hình ảnh sau khi resize về kích thước 115x100", use_column_width=True)

        restore_img = func_GFPGAN(input_img=cropped_image, upscale=6)

        image_copy = restore_img.copy()
        # st.image(cropped_image, caption="Hình ảnh sau khi khôi phục", use_column_width=True)
    else:
        image_copy = cv2.resize(cropped_image, (690, 600), interpolation = cv2.INTER_AREA)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)
    # Use canny edge detection
    edges = cv2.Canny(gray,100,200,apertureSize=3)
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                5, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                # np.pi/120, # Angle resolution in radians
                threshold=200, # Min number of votes for valid line
                minLineLength=360, # Min allowed length of line
                maxLineGap= 40 # Max allowed gap between line for joining them
                )
    angle_deg = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg_check = np.degrees(angle_rad)
        # cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if angle_deg == 0:
            if y2 >= 0 and y2 < int(image_copy.shape[0]/2) and y1 >= 0 and y1 < int(image_copy.shape[0]/2) and np.abs(angle_deg_check) < 60:
                # cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
            if y2 > int(image_copy.shape[0]/2) and y2 < int(image_copy.shape[0]) and y1 >  int(image_copy.shape[0]/2) and y1 < int(image_copy.shape[0]) and np.abs(angle_deg_check) < 60:
                # cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
        else:
            # if y2 >= 0 and y2 < int(image_copy.shape[0]/2) and y1 >= 0 and y1 < int(image_copy.shape[0]/2) and np.abs(angle_deg_check) < 60 and np.abs(angle_deg_check) < angle_deg:
            #     cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     angle_rad = np.arctan2(y2 - y1, x2 - x1)
            #     angle_deg = np.degrees(angle_rad)
            if y2 > int(image_copy.shape[0]/2) and y2 < int(image_copy.shape[0]) and y1 >  int(image_copy.shape[0]/2) and y1 < int(image_copy.shape[0]) and np.abs(angle_deg_check) < 60 and np.abs(angle_deg_check) < angle_deg:
                # cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)

    rotated_image = imutils.rotate(image_copy, angle_deg)

    gray = cv2.cvtColor(rotated_image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200,apertureSize=3)
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                5, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                # np.pi/120, # Angle resolution in radians
                threshold=200, # Min number of votes for valid line
                minLineLength=360, # Min allowed length of line
                maxLineGap= 50 # Max allowed gap between line for joining them
                )
    angle_deg = 0
    distance_top = 600
    distance_bottom = 600
    y_min, y_max = 0, 0
    deg = 0
    # height_tmp = int(image.shape[1])
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 125), 2)
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        if y_min == 0:
            if y2 < int(rotated_image.shape[0]/2) and y1 < int(rotated_image.shape[0]/2) and are_lines_parallel(angle_deg, threshold=3) and y_min < y2:
                # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                y_min = y2
        else:
            if y2 < int(rotated_image.shape[0]/2) and y1 < int(rotated_image.shape[0]/2) and are_lines_parallel(angle_deg, threshold=3) and y_min > y2:
                # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                y_min = y2
        if y2 > int(rotated_image.shape[0]/2) and y1 > int(rotated_image.shape[0]/2) and are_lines_parallel(angle_deg, threshold=4) and y_max < y2 and y2 > rotated_image.shape[0]/2 + 50:
            # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            y_max = y2

        # if x1 < int(rotated_image.shape[1]/2) and x2 < int(rotated_image.shape[1]/2):
        # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        if are_lines_perpendicular(angle_deg, threshold=2) == False and np.abs(angle_deg) > 45 and np.abs(angle_deg) > deg:
            if x1 <= x2 and y1 > y2 and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                tmp = int(np.tan(np.deg2rad(np.abs(np.abs(angle_deg) - 90))) * np.abs(y1 - y2)) 
                if tmp <= distance_top :
                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    distance_top = tmp
                    deg = np.abs(angle_deg)
            elif x1 <= x2 and y1 < y2 and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                tmp = int(np.tan(np.deg2rad(np.abs(np.abs(angle_deg) - 90))) * np.abs(y1 - y2)) 
                if tmp <= distance_bottom:
                    distance_bottom = tmp
                    deg = np.abs(angle_deg)

        elif are_lines_perpendicular(angle_deg) and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
            # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            distance_bottom = 0
            distance_top = 0
            deg = 0
    if(y_max == 0):
        y_max = rotated_image.shape[0]

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
                    5, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    # threshold=100, # Min number of votes for valid line
                    threshold=110, # Min number of votes for valid line
                    minLineLength=300, # Min allowed length of line
                    maxLineGap= 35 # Max allowed gap between line for joining them
                    )
        distance_top = 0
        distance_bottom = 0
        x_min, y_min, x_max, y_max = 0, 0, rotated_image.shape[1],rotated_image[0]
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
        # st.image(rotated_image, caption='Hình ảnh với xoay', use_column_width=True)
        final_image = rotated_image[y_min:y_max, x_min:x_max]


        img_gray_lp = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

        LP_WIDTH = final_image.shape[1]
        LP_HEIGHT = final_image.shape[0]
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
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height and intY > LP_HEIGHT/3 :
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

            if (predicted_class[0]) >= 10:
                character.append(chr((predicted_class[0] - 10) + ord('A')))
            else:
                character.append(predicted_class[0])

        char_array = [str(item) for item in character]
        result_string = ''.join(char_array[:])
        list_result = []
        list_result.append(result_string)
        # if len(result_string) == 0:
        #     s = f"<p style='font-size:100px; text-align: center'>🥺</p>"
        #     st.markdown(s, unsafe_allow_html=True) 
        if len(result_string) >= 0 and len(result_string) < 9:
            try:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 150)
                list_result.append(result_string)
                if len(result_string) >=0 and len(result_string) < 9:
                    img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 180)
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 130)
                            list_result.append(result_string)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            except:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                list_result.append(result_string)

        elif len(result_string) == 9:
            list_result.append(result_string)
        else:
            try:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 150)
                list_result.append(result_string)
                if len(result_string) >=0 and len(result_string) < 9:
                    img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 180)
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 130)
                            list_result.append(result_string)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            except:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                list_result.append(result_string)
    else:
        gray = cv2.cvtColor(warp_dst,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,250,apertureSize=3)
        lines = cv2.HoughLinesP(
                    edges, # Input edge image
                    5, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    # threshold=100, # Min number of votes for valid line
                    threshold=110, # Min number of votes for valid line
                    minLineLength=300, # Min allowed length of line
                    maxLineGap= 35 # Max allowed gap between line for joining them
                    )
        x_min, y_min, x_max, y_max = 0,0,warp_dst.shape[1],warp_dst.shape[0]
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(warp_dst, (x1, y1), (x2, y2), (255, 255, 0), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
                if are_lines_perpendicular(angle_deg, threshold=4) and x1 < int(warp_dst.shape[1]/2) and x1 > 10 and x2 > 10:
                    # cv2.line(warp_dst, (x1, y1), (x2, y2), (0, 255, 125), 2)
                    if x_min != 0 and x1 < x_min:
                        x_min = int((x1 + x2)/2)
                    elif x_min == 0:
                        x_min = int((x1 + x2)/2)
                if are_lines_perpendicular(angle_deg, threshold=4) and x1 > int(warp_dst.shape[1]/2) and x1 < warp_dst.shape[1]-10 and x2 < warp_dst.shape[1]-10:
                    # cv2.line(warp_dst, (x1, y1), (x2, y2), (125, 255, 0), 2)
                    if x_max != warp_dst.shape[1] and x1 > x_max:
                        x_max = int((x1 + x2)/2)
                    elif x_max == warp_dst.shape[1]:
                        x_max = int((x1 + x2)/2)
        except:
            pass
        final_image = warp_dst[:, x_min:x_max]

        img_gray_lp = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

        LP_WIDTH = final_image.shape[1]
        LP_HEIGHT = final_image.shape[0]

        #estimations of character contours sizes of cropped license plates
        dimensions = [LP_WIDTH/14,
                            LP_WIDTH/4,
                            LP_HEIGHT/3,
                            LP_HEIGHT/2]

        _, img_binary_lp = cv2.threshold(img_gray_lp, 170, 255, cv2.THRESH_BINARY)
        # _, img_binary_lp = cv2.threshold(img_gray_lp, 150, 255, cv2.THRESH_BINARY)

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

        list_result = []
        list_result.append(result_string)
        # if len(result_string) == 0:
        #     s = f"<p style='font-size:100px; text-align: center'>🥺</p>"
        #     st.markdown(s, unsafe_allow_html=True) 
        if len(result_string) >= 0 and len(result_string) < 9:
            try:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 150)
                list_result.append(result_string)
                if len(result_string) >=0 and len(result_string) < 9:
                    img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 180)
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 130)
                            list_result.append(result_string)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            except:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                list_result.append(result_string)

        elif len(result_string) == 9:
            pass
        else:
            try:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 150)
                list_result.append(result_string)
                if len(result_string) >=0 and len(result_string) < 9:
                    img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 180)
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 130)
                            list_result.append(result_string)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            except:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                list_result.append(result_string)

    longest_string = max(list_result, key=len)
    return longest_string, xywhcc


def RunDemo(yolo, cnn, image, version = 2):
    img, img_draw, cropped_image = None, None, None
    image_real = image.copy()
    # image_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB

    # Thay đổi kích thước hình ảnh
    resized_image = cv2.resize(image_real, (416, 416), interpolation = cv2.INTER_AREA)
    image_pil = resized_image
    resized_image = resized_image.astype(float)  # Chuyển đổi kiểu dữ liệu thành số thực
    resized_image /= 255  # Chuẩn hóa giá trị pixel về khoảng từ 0 đến 1

    # Mở rộng kích thước của hình ảnh để tạo batch
    img = np.expand_dims(resized_image, axis=0)

    prediction = yolo.model.predict(img)
    if version == 2:
        xywhcp = decode(*prediction, class_num=num_classes, threshold=0.7, version=2)
    else:
        xywhcp = decode(prediction[2][0],prediction[1][0],prediction[0][0] , class_num=num_classes, threshold=0.5, version=2)

    if len(xywhcp) > 0 and version == 2:
        xywhcp = nms(xywhcp, num_classes, 0.7)
    elif len(xywhcp) > 0 and version == 3:
        xywhcp = soft_nms(xywhcp, num_classes, 0.5, version=version)
    elif len(xywhcp) > 0 and version == 4:
        xywhcp = soft_nms(xywhcp, num_classes, 0.75, version=version)

    # Tạo hình vẽ từ hình ảnh gốc
    img_draw = Image.fromarray(image)
    draw = ImageDraw.Draw(img_draw)

    xywhcc = []
    xywhcc.append(xywhcp[0][0] * image.shape[1])
    xywhcc.append(xywhcp[0][1] * image.shape[0])
    xywhcc.append(xywhcp[0][2] * image.shape[1])
    xywhcc.append(xywhcp[0][3] * image.shape[0])
    xywhcc.append(xywhcp[0][4])
    xywhcc.append(xywhcp[0][5])


    if version == 4:
        x = int(xywhcp[0][0] * image.shape[1])
        y = int(xywhcp[0][1] * image.shape[0])
        w = int(xywhcp[0][2] * image.shape[1]*1.5)
        h = int(xywhcp[0][3] * image.shape[0]*1.2)
    else:
        x = int(xywhcp[0][0] * image.shape[1])
        y = int(xywhcp[0][1] * image.shape[0])
        w = int(xywhcp[0][2] * image.shape[1] * 1.3)
        h = int(xywhcp[0][3] * image.shape[0] * 1.1)
    class_i = int(xywhcp[0][5])

    # Vẽ hình tròn
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')

    # Vẽ hình chữ nhật
    x_min, y_min = int(x - w / 2), int(y - h / 2)
    x_max, y_max = int(x + w / 2), int(y + h / 2)
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
        
    draw.rectangle([x_min, y_min, x_max, y_max], outline='red')

    cropped_image = image[y_min:y_max, x_min:x_max]
    if cropped_image.shape[0] >= 100:
        image_copy = cv2.resize(cropped_image, (690, 600), interpolation = cv2.INTER_AREA)
        image_cut = image_copy.copy()
    else:
        cropped_image = cv2.resize(cropped_image, (115, 100), interpolation = cv2.INTER_AREA)
        restore_img = func_GFPGAN(input_img=cropped_image, upscale=6)
        image_copy = restore_img.copy()
        image_cut = image_copy.copy()
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)
    # Use canny edge detection
    edges = cv2.Canny(gray,100,200,apertureSize=3)
    thresholds = [200, 220, 100, 50]
    rhos = [5,2]
    angle_deg = 0
    lineGaps = [40, ]
    for lineGap in lineGaps:
        for rho in rhos:
            for threshold in thresholds:
                lines = cv2.HoughLinesP(
                            edges, 
                            rho, 
                            np.pi/180,
                            threshold=threshold, 
                            minLineLength=360, 
                            maxLineGap= lineGap
                            )
                try:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle_rad = np.arctan2(y2 - y1, x2 - x1)
                        angle_deg_check = np.degrees(angle_rad)
                        # cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if angle_deg == 0:
                            if y2 >= 0 and y2 < int(image_copy.shape[0]/2) and y1 >= 0 and y1 < int(image_copy.shape[0]/2) and np.abs(angle_deg_check) < 60:
                                # cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                                angle_deg = np.degrees(angle_rad)
                            if y2 > int(image_copy.shape[0]/2) and y2 < int(image_copy.shape[0]) and y1 >  int(image_copy.shape[0]/2) and y1 < int(image_copy.shape[0]) and np.abs(angle_deg_check) < 60:
                                # cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                                angle_deg = np.degrees(angle_rad)
                        else:
                            # if y2 >= 0 and y2 < int(image_copy.shape[0]/2) and y1 >= 0 and y1 < int(image_copy.shape[0]/2) and np.abs(angle_deg_check) < 60 and np.abs(angle_deg_check) < angle_deg:
                            #     cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            #     angle_rad = np.arctan2(y2 - y1, x2 - x1)
                            #     angle_deg = np.degrees(angle_rad)
                            if y2 > int(image_copy.shape[0]/2) and y2 < int(image_copy.shape[0]) and y1 > int(image_copy.shape[0]/2) and y1 < int(image_copy.shape[0]) and np.abs(angle_deg_check) < 60 and np.abs(angle_deg_check) < angle_deg:
                                # cv2.line(image_cut, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                                angle_deg = np.degrees(angle_rad)
                except:
                    pass
    rotated_image = imutils.rotate(image_copy, angle_deg)
    thresholds = [135, 100, 80, 150, 50]
    rhos = [1, 2, 3]
    lineGaps = [35, 50]
    y_min, y_max = 0, 0
    for lineGap in lineGaps:
        for rho in rhos:
            for threshold in thresholds:
                gray = cv2.cvtColor(rotated_image,cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray,50,200,apertureSize=3)
                lines = cv2.HoughLinesP(
                                    edges, 
                                    rho, 
                                    # np.pi/180, 
                                    np.pi/120, 
                                    threshold=threshold,
                                    minLineLength=300, 
                                    maxLineGap= lineGap
                                    )
                
                distance_top = 600
                distance_bottom = 600
                # y_min, y_max = 0, 0
                angle_deg = 0
                deg = 0
                # height_tmp = int(image.shape[1])
                try:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 125), 2)
                        angle_rad = np.arctan2(y2 - y1, x2 - x1)
                        angle_deg = np.degrees(angle_rad)
                        if y_min == 0:
                            if y2 < int(rotated_image.shape[0]/2) - int(rotated_image.shape[0]/6) and y1 < int(rotated_image.shape[0]/2) - int(rotated_image.shape[0]/6) and are_lines_parallel(angle_deg, threshold=5) and y_min < y2 and y2 > 10:
                                y_min = y2
                        else:
                            if y2 < int(rotated_image.shape[0]/2) - 10 and y1 < int(rotated_image.shape[0]/2) - 10 and are_lines_parallel(angle_deg, threshold=5) and y_min > y2 and y2 > 10:
                                y_min = y2
                        if y2 > int(rotated_image.shape[0]/2) and y1 > int(rotated_image.shape[0]/2) and are_lines_parallel(angle_deg, threshold=5) and y_max < y2 and y2 > rotated_image.shape[0]/2 + 50 and y2 < rotated_image.shape[0] - 20:
                            y_max = y2

                        angle_rad = np.arctan2(y2 - y1, x2 - x1)
                        angle_deg = np.degrees(angle_rad)
                        if are_lines_perpendicular(angle_deg, threshold=2) == False and np.abs(angle_deg) > 45 and np.abs(angle_deg) > deg:
                            if x1 <= x2 and y1 > y2 and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                                if x1 > int(rotated_image.shape[1]/2 + rotated_image.shape[1]/6) or x1 < int(rotated_image.shape[1]/2 - rotated_image.shape[1]/6):
                                    tmp = int(np.tan(np.deg2rad(np.abs(np.abs(angle_deg) - 90))) * np.abs(y1 - y2)) 
                                    if tmp <= distance_top :
                                        distance_top = tmp
                                        deg = np.abs(angle_deg)
                            elif x1 <= x2 and y1 < y2 and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                                if x1 > int(rotated_image.shape[1]/2 + rotated_image.shape[1]/6) or x1 < int(rotated_image.shape[1]/2 - rotated_image.shape[1]/6):
                                    tmp = int(np.tan(np.deg2rad(np.abs(np.abs(angle_deg) - 90))) * np.abs(y1 - y2)) 
                                    if tmp <= distance_bottom:
                                        distance_bottom = tmp
                                        deg = np.abs(angle_deg)
                        
                        elif are_lines_perpendicular(angle_deg) and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                            if x1 > int(rotated_image.shape[1]/2 + rotated_image.shape[1]/3) or x1 < int(rotated_image.shape[1]/2 - rotated_image.shape[1]/3):
                                # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                distance_bottom = 0
                                distance_top = 0
                                deg = 0

                except:
                    pass 
                # if(y_max == 0):
                #     y_max = rotated_image.shape[0]


                if y_min != 0 and y_max != 0 and distance_bottom != 600:
                    # st.write('đủ điều kiện ' + str(threshold))
                    break
                if y_min != 0 and y_max != 0 and distance_top != 600:
                    # st.write('đủ điều kiện ' + str(threshold))
                    break
            if y_min != 0 and y_max != 0 and distance_bottom != 600 and distance_top != 600:
                # st.write('đủ điều kiện ' + str(threshold))
                break
        if y_min != 0 and y_max != 0 and distance_bottom != 600:
            # st.write('đủ điều kiện ' + str(threshold))
            break
        if y_min != 0 and y_max != 0 and distance_top != 600:
            # st.write('đủ điều kiện ' + str(threshold))
            break

    # st.image(rotated_image, caption='Hình ảnh với hình xoay', use_column_width=True)
    if(y_max == 0 and y_min ==0):
        y_min = 0
        y_max = rotated_image.shape[0]
    if y_max == 0:
        y_max = rotated_image.shape[0]

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
        thresholds = [135, 100, 80, 150, 50]
        for threshold in thresholds:
            lines = cv2.HoughLinesP(
                            edges, 
                            1, 
                            # np.pi/180, 
                            np.pi/120, 
                            threshold=threshold,
                            minLineLength=200, 
                            maxLineGap=100
                            )
            x_min, x_max = 0, rotated_image.shape[1]
            flag_left, flag_right = False, False
            deg = 0
            # height_tmp = int(image.shape[1])
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 125), 2)
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)


                # if are_lines_parallel(angle_deg) and y1 < int(rotated_image.shape[0]/2):
                #     # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                #     y_min = y1
                # if are_lines_parallel(angle_deg) and y1 > int(rotated_image.shape[0]/2):
                #     # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                #     y_max = y1
                if are_lines_perpendicular(angle_deg) and x1 < int(rotated_image.shape[1]/2):
                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 125), 2)
                    flag_left = True
                    x_min = int((x1 + x2)/2)
                if are_lines_perpendicular(angle_deg) and x1 > int(rotated_image.shape[1]/2):
                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (125, 255, 0), 2)
                    flag_right = True
                    x_max = int((x1 + x2)/2)

            if flag_left and flag_right:
                    break
        final_image = rotated_image[y_min:y_max, x_min:x_max]


        img_gray_lp = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

        LP_WIDTH = final_image.shape[1]
        LP_HEIGHT = final_image.shape[0]
        dimensions = [LP_WIDTH/14,
                            LP_WIDTH/4,
                            LP_HEIGHT/4,
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
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height and intY > LP_HEIGHT/3 :
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
        else:
            img_res = []
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
        list_result = []
        list_result.append(result_string)
        # if len(result_string) == 0:
        #     s = f"<p style='font-size:100px; text-align: center'>🥺</p>"
        #     st.markdown(s, unsafe_allow_html=True) 
        if len(result_string) >= 0 and len(result_string) < 9:
            try:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 150)
                list_result.append(result_string)
                if len(result_string) >=0 and len(result_string) < 9:
                    img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 180)
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 130)
                            list_result.append(result_string)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            except:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                list_result.append(result_string)

        elif len(result_string) == 9:
            pass
        else:
            try:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 150)
                list_result.append(result_string)
                if len(result_string) >=0 and len(result_string) < 9:
                    img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 180)
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 130)
                            list_result.append(result_string)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            except:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                list_result.append(result_string)
    else:
        gray = cv2.cvtColor(warp_dst,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,200,apertureSize=3)
        thresholds = [135, 100, 80, 150, 50]
        rhos = [1,2]
        for rho in rhos:
            for threshold in thresholds:
                lines = cv2.HoughLinesP(
                                edges, 
                                rho, 
                                # np.pi/180, 
                                np.pi/120, 
                                threshold=threshold,
                                minLineLength=200, 
                                maxLineGap= 40
                                )
                x_min, y_min, x_max, y_max = 0,0,warp_dst.shape[1],warp_dst.shape[0]
                flag_left, flag_right = False, False
                try:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # cv2.line(warp_dst, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        angle_rad = np.arctan2(y2 - y1, x2 - x1)
                        angle_deg = np.degrees(angle_rad)
                        if are_lines_perpendicular(angle_deg, threshold=8) and x1 < int(warp_dst.shape[1]/2) - int(warp_dst.shape[1]/4) and x1 > 10 and x2 > 10:
                            flag_left = True
                            # cv2.line(warp_dst, (x1, y1), (x2, y2), (0, 255, 125), 2)
                            if x_min != 0 and x1 < x_min:
                                x_min = int((x1 + x2)/2) - 10
                            elif x_min == 0:
                                x_min = int((x1 + x2)/2) - 10
                        if are_lines_perpendicular(angle_deg, threshold=6) and x1 > int(warp_dst.shape[1]/2) + int(warp_dst.shape[1]/4) and x1 < warp_dst.shape[1]-10 and x2 < warp_dst.shape[1]-10:
                            flag_right = True
                            # cv2.line(warp_dst, (x1, y1), (x2, y2), (125, 255, 0), 2)
                            if x_max != warp_dst.shape[1] and x1 > x_max:
                                x_max = int((x1 + x2)/2)
                            elif x_max == warp_dst.shape[1]:
                                x_max = int((x1 + x2)/2)
                except:
                    pass

                if flag_left and flag_right:
                    break
            if flag_left and flag_right:
                break
        # st.image(warp_dst, caption='Hình ảnh warp_dst', use_column_width=True)
        final_image = warp_dst[:, x_min:x_max]
        # st.image(cropped_image, caption='Hình ảnh crop cuối', use_column_width=True)


        img_gray_lp = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)

        LP_WIDTH = final_image.shape[1]
        LP_HEIGHT = final_image.shape[0]

        #estimations of character contours sizes of cropped license plates
        dimensions = [LP_WIDTH/14,
                            LP_WIDTH/4,
                            LP_HEIGHT/4,
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
            if intX < 0:
                intX = 1
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
        else:
            img_res = []
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
        list_result = []
        list_result.append(result_string)
        # if len(result_string) == 0:
        #     s = f"<p style='font-size:100px; text-align: center'>🥺</p>"
        #     st.markdown(s, unsafe_allow_html=True) 
        if len(result_string) >= 0 and len(result_string) < 9:
            try:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 150)
                list_result.append(result_string)
                if len(result_string) >=0 and len(result_string) < 9:
                    img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 180)
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 130)
                            list_result.append(result_string)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            except:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                list_result.append(result_string)

        elif len(result_string) == 9:
            pass
        else:
            try:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 150)
                list_result.append(result_string)
                if len(result_string) >=0 and len(result_string) < 9:
                    img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 180)
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 130)
                            list_result.append(result_string)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            except:
                img_binary_lp, result_string = Rerun(final_image, cnn, threshold = 190)
                list_result.append(result_string)

    longest_string = max(list_result, key=len)
    return longest_string, xywhcc

def RunPredictionV8(yolo, img):
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

    image_real = img.copy()
    
    orig_h, orig_w, _ = image_real.shape
    img_transform, _, ratio = val_transform(image_real)
    img_transform = img_transform.unsqueeze(0).to(device)

    # inference
    outputs = yolo(img_transform)
    scores = outputs['scores']
    labels = outputs['labels']
    bboxes = outputs['bboxes']
    index = np.argmax(scores)

    bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], ratio)
    x = int((int(bboxes[index][0]) + int(bboxes[index][2]))/2)
    y = int((int(bboxes[index][1]) + int(bboxes[index][3]))/2) 
    w = int(bboxes[index][2] - bboxes[index][0])
    h = int(bboxes[index][3] - bboxes[index][1])

    xywhcc = []
    xywhcc.append(x)
    xywhcc.append(y)
    xywhcc.append(w)
    xywhcc.append(h)
    xywhcc.append(scores[0])
    xywhcc.append(labels[0])

    return xywhcc
def RunPrediction(yolo, image, version = 2):
    img, img_draw, cropped_image = None, None, None
    image_real = image.copy()
    # image_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB

    # Thay đổi kích thước hình ảnh
    resized_image = cv2.resize(image_real, (416, 416), interpolation = cv2.INTER_AREA)
    resized_image = resized_image.astype(float)  # Chuyển đổi kiểu dữ liệu thành số thực
    resized_image /= 255  # Chuẩn hóa giá trị pixel về khoảng từ 0 đến 1

    img = np.expand_dims(resized_image, axis=0)

    prediction = yolo.model.predict(img)
    if version == 2:
        xywhcp = decode(*prediction, class_num=num_classes, threshold=0.7, version=2)
    else:
        xywhcp = decode(prediction[2][0],prediction[1][0],prediction[0][0] , class_num=num_classes, threshold=0.5, version=2)

    if len(xywhcp) > 0 and version == 2:
        xywhcp = nms(xywhcp, num_classes, 0.7)
        # xywhcp = soft_nms(xywhcp, num_classes, 0.5, version=version)
    elif len(xywhcp) > 0 and version == 3:
        xywhcp = soft_nms(xywhcp, num_classes, 0.5, version=version)
    elif len(xywhcp) > 0 and version == 4:
        xywhcp = soft_nms(xywhcp, num_classes, 0.75, version=version)

    xywhcc = []
    xywhcc.append(xywhcp[0][0] * image.shape[1])
    xywhcc.append(xywhcp[0][1] * image.shape[0])
    xywhcc.append(xywhcp[0][2] * image.shape[1])
    xywhcc.append(xywhcp[0][3] * image.shape[0])
    xywhcc.append(xywhcp[0][4])
    xywhcc.append(xywhcp[0][5])

    return xywhcc


with st.form("select_anchors"):
    path_test = st.text_input('Nhập đường dẫn tới thư mục chứa dữ liệu test', placeholder="Path....")

    # st.markdown(styled_s + style, unsafe_allow_html=True)
    select_model = st.selectbox("Chọn mô hình", ("Yolov2", "Yolov3", "Yolov4", "Yolov8"))
    start = st.form_submit_button('Bắt đầu')

if start:
    count = 0
    count_4_character = 0
    count_p = 0
    score_iou = 0.0
    score_giou = 0.0
    score_diou = 0.0
    conf = 0.0
    mAP = 0.0
    with open(path_test + '\\label.txt', 'r') as file:
        total = len(file.readlines())
    if(select_model == 'Yolov2'):
        contents = [file for file in os.listdir(path_test) if file.endswith(".jpg")]
        start_time = time.time()
        for item in contents:
            file_name = path_test + "\\" + item
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                plate_number, xywhcc = RunDemo(st.session_state.yolov2, st.session_state.cnn, image, version=2)
                conf += float(xywhcc[4])
                p = int(xywhcc[5])
                if p == 0:
                    count_p += 1
                with open(path_test + '\\label.txt', 'r') as file:
                    for line in file:
                        data = line.strip().split()
                        name, plate_small, plate  = data[0], data[1], data[2]
                        if name == item:
                            xywh_true = [int((int(data[5]) + int(data[3]))/2), int((int(data[6]) + int(data[4]))/2), int(int(data[5]) - int(data[3])), int(int(data[6]) - int(data[4]))]
                            xywh_pred = [int(xywhcc[0]), int(xywhcc[1]), int(xywhcc[2]), int(xywhcc[3])]
                            score_iou += iou(xywh_true, xywh_pred)
                            score_giou += giou(xywh_true, xywh_pred)
                            score_diou += diou(xywh_true, xywh_pred)
                        if name == item and plate_number == plate:
                            count+=1
                        if name == item and plate_number[:4] == plate_small:
                            count_4_character+=1
            except:
                pass
        end_time = time.time()

        list_recall = []
        list_precision = []
        list_iou_all = []
        list_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # list_iou = [0.5]
        for i in range(len(list_iou)):
            TP, FP, FN = 0,0,0
            for item in contents:
                file_name = path_test + "\\" + item
                image = cv2.imread(file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                try:
                    xywhcc = RunPrediction(st.session_state.yolov2, image, version=2)
                    with open(path_test + '\\label.txt', 'r') as file:
                        for line in file:
                            data = line.strip().split()
                            name, plate_small, plate  = data[0], data[1], data[2]
                            if name == item:
                                print(name)
                                xywh_true = [int((int(data[5]) + int(data[3]))/2), int((int(data[6]) + int(data[4]))/2), int(int(data[5]) - int(data[3])), int(int(data[6]) - int(data[4]))]
                                xywh_pred = [int(xywhcc[0]), int(xywhcc[1]), int(xywhcc[2]), int(xywhcc[3])]
                                _iou = iou(xywh_true, xywh_pred)
                                if _iou > 0:
                                    list_iou_all.append(_iou)
                                if _iou >= list_iou[i]:
                                    TP += 1
                                # elif _iou < list_iou[i] and _iou >= list_iou[i] - 0.5:
                                elif _iou < list_iou[i] and _iou > 0:
                                    FP += 1
                                else:
                                    FN += 1
                except: 
                    FN += 1
            try:
                if (TP + FN) == 0:
                    recall = 0.0
                else:
                    recall = float(((TP)/(TP + FN)))
                if(TP + FP) == 0:
                    precision = 0.0
                else:
                    precision = float(((TP)/(TP + FP)))
                list_recall.append(recall)
                list_precision.append(precision)
            except:
                pass

        AP = 0.0
        list_recall = sorted(list_recall)
        print(list_iou_all)
        np_list_precision = np.array(list_precision)
        np_list_recall = np.array(list_recall)
        AP = np.sum(np_list_precision)
        AP = round(AP / len(list_recall), 2)  
        # if all_zero_or_one:
        #     AP = np.sum(np_list_precision)
        #     AP = round(AP / len(list_recall), 2)  
        # else:
        #     if is_all_ones:
        #         AP = np.sum(np_list_precision)
        #     elif is_all_zero:
        #         AP = 0.0
        #     else:
        #         for i in range(1, len(list_recall)):
        #             AP += (np.abs(np_list_recall[i] - np_list_recall[i - 1]))*np_list_precision[i]
        #     try:
        #         if is_all_ones or is_all_zero:
        #             AP = round(AP / len(list_recall), 2)    
        #         else:
        #             AP = round(AP, 2)
        #     except:
        #         pass

        
        interval = end_time - start_time
        hours = int(interval // 3600)
        minutes = int((interval % 3600) // 60)
        seconds = int(interval % 60)
    if(select_model == 'Yolov3'):
        contents = [file for file in os.listdir(path_test) if file.endswith(".jpg")]
        start_time = time.time()
        for item in contents:
            file_name = path_test + "\\" + item
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                plate_number, xywhcc = RunDemo(st.session_state.yolov3, st.session_state.cnn, image, version=3)
                conf += float(xywhcc[4])
                p = int(xywhcc[5])
                if p == 0:
                    count_p += 1
                with open(path_test + '\\label.txt', 'r') as file:
                    for line in file:
                        data = line.strip().split()
                        name, plate_small, plate  = data[0], data[1], data[2]
                        if name == item:
                            xywh_true = [int((int(data[5]) + int(data[3]))/2), int((int(data[6]) + int(data[4]))/2), int(int(data[5]) - int(data[3])), int(int(data[6]) - int(data[4]))]
                            xywh_pred = [int(xywhcc[0]), int(xywhcc[1]), int(xywhcc[2]), int(xywhcc[3])]
                            score_iou += iou(xywh_true, xywh_pred)
                            score_giou += giou(xywh_true, xywh_pred)
                            score_diou += diou(xywh_true, xywh_pred)
                        if name == item and plate_number == plate:
                            count+=1
                        if name == item and plate_number[:4] == plate_small:
                            count_4_character+=1
            except:
                pass
        end_time = time.time()
        
        list_iou_all = []
        list_recall = []
        list_precision = []
        list_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # list_iou = [0.5]
        for i in range(len(list_iou)):
            TP, FP, FN = 0,0,0
            for item in contents:
                file_name = path_test + "\\" + item
                image = cv2.imread(file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                try:
                    xywhcc = RunPrediction(st.session_state.yolov3, image, version=3)
                    with open(path_test + '\\label.txt', 'r') as file:
                        for line in file:
                            data = line.strip().split()
                            name, plate_small, plate  = data[0], data[1], data[2]
                            if name == item:
                                xywh_true = [int((int(data[5]) + int(data[3]))/2), int((int(data[6]) + int(data[4]))/2), int(int(data[5]) - int(data[3])), int(int(data[6]) - int(data[4]))]
                                xywh_pred = [int(xywhcc[0]), int(xywhcc[1]), int(xywhcc[2]), int(xywhcc[3])]
                                _iou = iou(xywh_true, xywh_pred)
                                if _iou > 0:
                                    list_iou_all.append(_iou)
                                if _iou >= list_iou[i]:
                                    TP += 1
                                # elif _iou < list_iou[i] and _iou >= list_iou[i] - 0.5:
                                elif _iou < list_iou[i] and _iou > 0:
                                    FP += 1
                                else:
                                    FN += 1
                except: 
                    FN += 1
            try:
                if (TP + FN) == 0:
                    recall = 0.0
                else:
                    recall = float(((TP)/(TP + FN)))
                if(TP + FP) == 0:
                    precision = 0.0
                else:
                    precision = float(((TP)/(TP + FP)))
                list_recall.append(recall)
                list_precision.append(precision)
            except:
                pass

        AP = 0.0
        list_recall = sorted(list_recall)
        np_list_precision = np.array(list_precision)
        np_list_recall = np.array(list_recall)
        AP = np.sum(np_list_precision)
        AP = round(AP / len(list_recall), 2)  

        interval = end_time - start_time
        hours = int(interval // 3600)
        minutes = int((interval % 3600) // 60)
        seconds = int(interval % 60)
    if(select_model == 'Yolov4'):
        contents = [file for file in os.listdir(path_test) if file.endswith(".jpg")]
        start_time = time.time()
        for item in contents:
            file_name = path_test + "\\" + item
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                plate_number, xywhcc = RunDemo(st.session_state.yolov4, st.session_state.cnn, image, version=4)
                conf += float(xywhcc[4])
                p = int(xywhcc[5])
                if p == 0:
                    count_p += 1
                with open(path_test + '\\label.txt', 'r') as file:
                    for line in file:
                        data = line.strip().split()
                        name, plate_small, plate  = data[0], data[1], data[2]
                        if name == item:
                            xywh_true = [int((int(data[5]) + int(data[3]))/2), int((int(data[6]) + int(data[4]))/2), int(int(data[5]) - int(data[3])), int(int(data[6]) - int(data[4]))]
                            xywh_pred = [int(xywhcc[0]), int(xywhcc[1]), int(xywhcc[2]), int(xywhcc[3])]
                            score_iou += iou(xywh_true, xywh_pred)
                            score_giou += giou(xywh_true, xywh_pred)
                            score_diou += diou(xywh_true, xywh_pred)
                        if name == item and plate_number == plate:
                            count+=1
                        if name == item and plate_number[:4] == plate_small:
                            count_4_character+=1
            except: 
                pass

        end_time = time.time()

        list_recall = []
        list_precision = []
        list_iou_all = []
        list_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # list_iou = [0.5]
        for i in range(len(list_iou)):
            TP, FP, FN = 0,0,0
            for item in contents:
                file_name = path_test + "\\" + item
                image = cv2.imread(file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                try:
                    xywhcc = RunPrediction(st.session_state.yolov4, image, version=4)
                    with open(path_test + '\\label.txt', 'r') as file:
                        for line in file:
                            data = line.strip().split()
                            name, plate_small, plate  = data[0], data[1], data[2]
                            if name == item:
                                xywh_true = [int((int(data[5]) + int(data[3]))/2), int((int(data[6]) + int(data[4]))/2), int(int(data[5]) - int(data[3])), int(int(data[6]) - int(data[4]))]
                                xywh_pred = [int(xywhcc[0]), int(xywhcc[1]), int(xywhcc[2]), int(xywhcc[3])]
                                _iou = iou(xywh_true, xywh_pred)
                                if _iou > 0:
                                    list_iou_all.append(_iou)
                                if _iou >= list_iou[i]:
                                    TP += 1
                                # elif _iou < list_iou[i] and _iou >= list_iou[i] - 0.5:
                                elif _iou < list_iou[i] and _iou > 0:
                                    FP += 1
                                else:
                                    FN += 1
                except: 
                    FN += 1
            try:
                if (TP + FN) == 0:
                    recall = 0.0
                else:
                    recall = float(((TP)/(TP + FN)))
                if(TP + FP) == 0:
                    precision = 0.0
                else:
                    precision = float(((TP)/(TP + FP)))
                list_recall.append(recall)
                list_precision.append(precision)
            except:
                pass

        AP = 0.0
        list_recall = sorted(list_recall)
        np_list_precision = np.array(list_precision)
        np_list_recall = np.array(list_recall)
        AP = np.sum(np_list_precision)
        AP = round(AP / len(list_recall), 2)  

        interval = end_time - start_time
        hours = int(interval // 3600)
        minutes = int((interval % 3600) // 60)
        seconds = int(interval % 60)

    if(select_model == 'Yolov8'):
        contents = [file for file in os.listdir(path_test) if file.endswith(".jpg")]
        start_time = time.time()
        for item in contents:
            file_name = path_test + "\\" + item
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                plate_number, xywhcc = RunDemoV8(st.session_state.yolov8, st.session_state.cnn, image)
                conf += float(xywhcc[4])
                p = int(xywhcc[5])
                if p == 0:
                    count_p += 1
                with open(path_test + '\\label.txt', 'r') as file:
                    for line in file:
                        data = line.strip().split()
                        name, plate_small, plate  = data[0], data[1], data[2]
                        if name == item:
                            xywh_true = [int((int(data[5]) + int(data[3]))/2), int((int(data[6]) + int(data[4]))/2), int(int(data[5]) - int(data[3])), int(int(data[6]) - int(data[4]))]
                            xywh_pred = [int(xywhcc[0]), int(xywhcc[1]), int(xywhcc[2]), int(xywhcc[3])]
                            score_iou += iou(xywh_true, xywh_pred)
                            score_giou += giou(xywh_true, xywh_pred)
                            score_diou += diou(xywh_true, xywh_pred)
                        if name == item and plate_number == plate:
                            count+=1
                        if name == item and plate_number[:4] == plate_small:
                            count_4_character+=1
            except: 
                pass
        end_time = time.time()
        list_iou_all = []
        list_recall = []
        list_precision = []
        list_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # list_iou = [0.5]
        for i in range(len(list_iou)):
            TP, FP, FN = 0,0,0
            for item in contents:
                file_name = path_test + "\\" + item
                image = cv2.imread(file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                try:
                    xywhcc = RunPredictionV8(st.session_state.yolov8, image)
                    with open(path_test + '\\label.txt', 'r') as file:
                        for line in file:
                            data = line.strip().split()
                            name, plate_small, plate  = data[0], data[1], data[2]
                            if name == item:
                                xywh_true = [int((int(data[5]) + int(data[3]))/2), int((int(data[6]) + int(data[4]))/2), int(int(data[5]) - int(data[3])), int(int(data[6]) - int(data[4]))]
                                xywh_pred = [int(xywhcc[0]), int(xywhcc[1]), int(xywhcc[2]), int(xywhcc[3])]
                                _iou = iou(xywh_true, xywh_pred)
                                if _iou > 0:
                                    list_iou_all.append(_iou)
                                if _iou >= list_iou[i]:
                                    TP += 1
                                # elif _iou < list_iou[i] and _iou >= list_iou[i] - 0.5:
                                elif _iou < list_iou[i] and _iou > 0:
                                    FP += 1
                                else:
                                    FN += 1
                except: 
                    FN += 1
            try:
                if (TP + FN) == 0:
                    recall = 0.0
                else:
                    recall = float(((TP)/(TP + FN)))
                if(TP + FP) == 0:
                    precision = 0.0
                else:
                    precision = float(((TP)/(TP + FP)))
                list_recall.append(recall)
                list_precision.append(precision)
            except:
                pass

        AP = 0.0
        list_recall = sorted(list_recall)
        np_list_precision = np.array(list_precision)
        np_list_recall = np.array(list_recall)

        is_all_ones = np.all(np_list_recall == 1.0)
        is_all_zero = np.all(np_list_recall == 0.0)
        all_zero_or_one = np.all((np_list_recall == 0.0) | (np_list_recall == 1.0))
        AP = np.sum(np_list_precision)
        AP = round(AP / len(list_recall), 2)  
        interval = end_time - start_time
        hours = int(interval // 3600)
        minutes = int((interval % 3600) // 60)
        seconds = int(interval % 60)

    result = round(count/total, 2)*100
    result_4 = round(count_4_character/total, 2)*100
    result_p = round(count_p/total, 2)*100
    # miou = round(score_iou / total, 2)
    # miou = round(score_iou / count_p, 2)
    miou = round(np.mean(list_iou_all), 2)
    mgiou = round(score_giou / total, 2)
    mdiou = round(score_diou / total, 2)
    conf = round(conf / total, 2) * 100
    s = f"<p style='font-size:40px;'>✅ mAP {AP}</p>"
    st.markdown(s, unsafe_allow_html=True) 
    s = f"<p style='font-size:40px;'>✅ Độ chính xác nhận diện 4 kí tự biển số xe trên tập Test là {result_4}%</p>"
    st.markdown(s, unsafe_allow_html=True) 
    s = f"<p style='font-size:40px;'>✅ Độ chính xác nhận diện kí tự biển số xe trên tập Test là {result}%</p>"
    st.markdown(s, unsafe_allow_html=True) 
    s = f"<p style='font-size:40px;'>✅ Độ chính xác nhận diện biển số xe trên tập Test là {result_p}%</p>"
    st.markdown(s, unsafe_allow_html=True) 
    s = f"<p style='font-size:40px;'>✅ Độ tự tin khi nhận điện biển số xe trên tập Test là {conf}%</p>"
    st.markdown(s, unsafe_allow_html=True) 
    s = f"<p style='font-size:40px;'>⏱️ Tổng thời gian Test: {hours} Giờ {minutes} Phút {seconds} Giây</p>"
    st.markdown(s, unsafe_allow_html=True) 
    s = r"$ mIoU = \frac{1}{N}\sum_{i=1}^{N} \frac{\text{Area\ of\ Intersection}}{\text{Area\ of\ Union}} $"
    styled_s = f"""<span class="markdown-css">{s}</span>"""
    style = """
    <style>
        .markdown-css {
            font-size: 30px;
            text-align: center;
            display: block;
        }
    </style>
    """
    st.markdown(styled_s + style, unsafe_allow_html=True)
    s = f"<p style='font-size:40px;'>✅ mIoU: {miou}</p>"
    st.markdown(s, unsafe_allow_html=True) 
    