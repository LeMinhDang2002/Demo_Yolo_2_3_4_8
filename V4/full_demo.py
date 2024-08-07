import streamlit as st
import cv2, numpy as np
from utils.kmeans import kmeans, iou_dist, euclidean_dist
from PIL import Image, ImageDraw
import asyncio, os
import imutils
import pandas as pd
import torch
from st_pages import add_page_title, hide_pages

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
    st.write(xywhcp_new)
    # xywhcp_new_list = xywhcp_new[0].tolist()

    # print(xywhcp_new[0])
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
    xywhcp_new = sorted(xywhcp_new[0], reverse=True, key=lambda x:x[4])
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

    return img_binary_lp, result_string


def DisplayDemo(yolo, cnn, uploaded_files, version = 2):
    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.write("Filename:", uploaded_file.name)
        st.write("Image shape:", img.shape)
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        image_real = img.copy()
        image_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB

        resized_image = cv2.resize(image_real, (416, 416), interpolation = cv2.INTER_AREA)
        image_pil = resized_image
        resized_image = resized_image.astype(float)  
        resized_image /= 255  

        image = np.expand_dims(resized_image, axis=0)

        prediction = yolo.model.predict(image)

        # x_min, y_min, x_max, y_max = display_img(img=image[0], prediction[2][0], prediction[1][0], prediction[0][0], conf_threshold=0.7)
        if version == 2:
            xywhcp = decode(*prediction, class_num=num_classes, threshold=0.7, version=2)
        else:
            xywhcp = decode(prediction[2][0],prediction[1][0],prediction[0][0] , class_num=num_classes, threshold=0.5, version=2)

        # xywhcp = nms(xywhcp, num_classes, 0.7)
        # print(xywhcp)
        if len(xywhcp) > 0 and version == 2:
            xywhcp = nms(xywhcp, num_classes, 0.7)
        elif len(xywhcp) > 0 and version == 3:
            xywhcp = soft_nms(xywhcp, num_classes, 0.5, version=version)
        elif len(xywhcp) > 0 and version == 4:
            xywhcp = soft_nms(xywhcp, num_classes, 0.75, version=version)

        # Tạo hình vẽ từ hình ảnh gốc
        img_draw = Image.fromarray(img)
        draw = ImageDraw.Draw(img_draw)

        if version == 4:
            x = int(xywhcp[0][0] * img.shape[1])
            y = int(xywhcp[0][1] * img.shape[0])
            w = int(xywhcp[0][2] * img.shape[1]*1.5)
            h = int(xywhcp[0][3] * img.shape[0]*1.2)
        else:
            x = int(xywhcp[0][0] * img.shape[1])
            y = int(xywhcp[0][1] * img.shape[0])
            w = int(xywhcp[0][2] * img.shape[1] * 1.3)
            h = int(xywhcp[0][3] * img.shape[0] * 1.1)
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

        # Hiển thị hình ảnh với hình vẽ
        st.image(img_draw, caption='Hình ảnh với hình vẽ', use_column_width=True)

        cropped_image = img[y_min:y_max, x_min:x_max]
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
        # st.image(image_cut, caption='Hình ảnh với hình vẽ cắt', use_column_width=True)
        rotated_image = imutils.rotate(image_copy, angle_deg)
        # st.image(rotated_image, caption='Hình ảnh với hình xoay', use_column_width=True)
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
                                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                    y_min = y2
                            else:
                                if y2 < int(rotated_image.shape[0]/2) - 10 and y1 < int(rotated_image.shape[0]/2) - 10 and are_lines_parallel(angle_deg, threshold=5) and y_min > y2 and y2 > 10:
                                    # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                    y_min = y2
                            if y2 > int(rotated_image.shape[0]/2) and y1 > int(rotated_image.shape[0]/2) and are_lines_parallel(angle_deg, threshold=5) and y_max < y2 and y2 > rotated_image.shape[0]/2 + 50 and y2 < rotated_image.shape[0] - 20:
                                # cv2.line(rotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                y_max = y2

                            # if x1 < int(rotated_image.shape[1]/2) and x2 < int(rotated_image.shape[1]/2):
                            # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            angle_rad = np.arctan2(y2 - y1, x2 - x1)
                            angle_deg = np.degrees(angle_rad)
                            if are_lines_perpendicular(angle_deg, threshold=2) == False and np.abs(angle_deg) > 45 and np.abs(angle_deg) > deg:
                                if x1 <= x2 and y1 > y2 and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                                    if x1 > int(rotated_image.shape[1]/2 + rotated_image.shape[1]/6) or x1 < int(rotated_image.shape[1]/2 - rotated_image.shape[1]/6):
                                        # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        tmp = int(np.tan(np.deg2rad(np.abs(np.abs(angle_deg) - 90))) * np.abs(y1 - y2)) 
                                        if tmp <= distance_top :
                                            distance_top = tmp
                                            deg = np.abs(angle_deg)
                                elif x1 <= x2 and y1 < y2 and x1 > 10 and x2>10 and x1 < rotated_image.shape[1] - 10 and x2 < rotated_image.shape[1] - 10:
                                    if x1 > int(rotated_image.shape[1]/2 + rotated_image.shape[1]/6) or x1 < int(rotated_image.shape[1]/2 - rotated_image.shape[1]/6):
                                        # cv2.line(rotated_image, (x1, y1), (x2, y2), (0, 200, 0), 2)
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

        # st.image(warp_dst, caption='Hình ảnh sau cropped', use_column_width=True)


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
            cropped_image = rotated_image[y_min:y_max, x_min:x_max]


            img_gray_lp = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            LP_WIDTH = cropped_image.shape[1]
            LP_HEIGHT = cropped_image.shape[0]
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
            if len(result_string) >= 0 and len(result_string) < 9:
                try:
                    img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 150)
                    list_result.append(result_string)
                    # st.image(img_binary_lp, caption='Hình ảnh nhị phân', use_column_width=True)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 180)
                        list_result.append(result_string)
                        # st.image(img_binary_lp, caption='Hình ảnh nhị phân 180', use_column_width=True)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 190)
                            list_result.append(result_string)
                            # st.image(img_binary_lp, caption='Hình ảnh nhị phân 190', use_column_width=True)
                            if len(result_string) >=0 and len(result_string) < 9:
                                img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 130)
                                list_result.append(result_string)
                                # st.image(img_binary_lp, caption='Hình ảnh nhị phân 130', use_column_width=True)
                                if(len(result_string) <8):
                                    longest_string = max(list_result, key=len)
                                    len_longest_string = len(longest_string)
                                    s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả kí tự. Chỉ có thể nhận diện được {len_longest_string} kí tự trên Bảng số xe</p>"
                                    st.markdown(s, unsafe_allow_html=True) 
                                    char = longest_string[2]
                                    if ord(char) >= 65 and ord(char) <= 90:
                                        s = f"<p style='font-size:50px;'>{longest_string[:2]}-{longest_string[2:4]} {longest_string[4:]}</p>"
                                        st.markdown(s, unsafe_allow_html=True)
                                        df = pd.read_excel('./BANG_SO_XE.xlsx')
                                        data_array = df.values
                                        for i in range(len(data_array)):
                                            if np.char.strip(data_array[i][1]) == longest_string[:4]:
                                                s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                                st.markdown(s, unsafe_allow_html=True)
                                                break
                                    else:
                                        s = f"<p style='font-size:50px;'>{longest_string}</p>"
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
                    # st.image(img_binary_lp, caption='Hình ảnh nhị phân 190', use_column_width=True)
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
                # st.image(img_binary_lp, caption='Hình ảnh nhị phân 170', use_column_width=True)
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
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 130)
                            list_result.append(result_string)
                            if(len(result_string) <8):
                                longest_string = max(list_result, key=len)
                                len_longest_string = len(longest_string)
                                s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả kí tự. Chỉ có thể nhận diện được {len_longest_string} kí tự trên Bảng số xe</p>"
                                st.markdown(s, unsafe_allow_html=True) 
                                char = longest_string[2]
                                if ord(char) >= 65 and ord(char) <= 90:
                                    s = f"<p style='font-size:50px;'>{longest_string[:2]}-{longest_string[2:4]} {longest_string[4:]}</p>"
                                    st.markdown(s, unsafe_allow_html=True)
                                    df = pd.read_excel('./BANG_SO_XE.xlsx')
                                    data_array = df.values
                                    for i in range(len(data_array)):
                                        if np.char.strip(data_array[i][1]) == longest_string[:4]:
                                            s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                            st.markdown(s, unsafe_allow_html=True)
                                            break
                                else:
                                    s = f"<p style='font-size:50px;'>{longest_string}</p>"
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
                    list_result.append(result_string)
                    if(len(result_string) <8):
                        longest_string = max(list_result, key=len)
                        len_longest_string = len(longest_string)
                        s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả kí tự. Chỉ có thể nhận diện được {len_longest_string} kí tự trên Bảng số xe</p>"
                        st.markdown(s, unsafe_allow_html=True) 
                        char = longest_string[2]
                        if ord(char) >= 65 and ord(char) <= 90:
                            s = f"<p style='font-size:50px;'>{longest_string[:2]}-{longest_string[2:4]} {longest_string[4:]}</p>"
                            st.markdown(s, unsafe_allow_html=True)
                            df = pd.read_excel('./BANG_SO_XE.xlsx')
                            data_array = df.values
                            for i in range(len(data_array)):
                                if np.char.strip(data_array[i][1]) == longest_string[:4]:
                                    s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                    st.markdown(s, unsafe_allow_html=True)
                                    break
                        else:
                            s = f"<p style='font-size:50px;'>{longest_string}</p>"
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
            cropped_image = warp_dst[:, x_min:x_max]
            st.image(cropped_image, caption='Hình ảnh crop cuối', use_column_width=True)


            img_gray_lp = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            LP_WIDTH = cropped_image.shape[1]
            LP_HEIGHT = cropped_image.shape[0]

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
                if intX < 0:
                    intX = 1
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
                    img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 150)
                    list_result.append(result_string)
                    # st.image(img_binary_lp, caption='Hình ảnh nhị phân', use_column_width=True)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 180)
                        list_result.append(result_string)
                        # st.image(img_binary_lp, caption='Hình ảnh nhị phân 180', use_column_width=True)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 190)
                            list_result.append(result_string)
                            # st.image(img_binary_lp, caption='Hình ảnh nhị phân 190', use_column_width=True)
                            if len(result_string) >=0 and len(result_string) < 9:
                                img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 130)
                                list_result.append(result_string)
                                # st.image(img_binary_lp, caption='Hình ảnh nhị phân 130', use_column_width=True)
                                if(len(result_string) <8):
                                    longest_string = max(list_result, key=len)
                                    len_longest_string = len(longest_string)
                                    s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả kí tự. Chỉ có thể nhận diện được {len_longest_string} kí tự trên Bảng số xe</p>"
                                    st.markdown(s, unsafe_allow_html=True) 
                                    char = longest_string[2]
                                    if ord(char) >= 65 and ord(char) <= 90:
                                        s = f"<p style='font-size:50px;'>{longest_string[:2]}-{longest_string[2:4]} {longest_string[4:]}</p>"
                                        st.markdown(s, unsafe_allow_html=True)
                                        df = pd.read_excel('./BANG_SO_XE.xlsx')
                                        data_array = df.values
                                        for i in range(len(data_array)):
                                            if np.char.strip(data_array[i][1]) == longest_string[:4]:
                                                s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                                st.markdown(s, unsafe_allow_html=True)
                                                break
                                    else:
                                        s = f"<p style='font-size:50px;'>{longest_string}</p>"
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
                    # st.image(img_binary_lp, caption='Hình ảnh nhị phân 190', use_column_width=True)
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
                # st.image(img_binary_lp, caption='Hình ảnh nhị phân', use_column_width=True)
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
                    list_result.append(result_string)
                    if len(result_string) >=0 and len(result_string) < 9:
                        img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 190)
                        list_result.append(result_string)
                        if len(result_string) >=0 and len(result_string) < 9:
                            img_binary_lp, result_string = Rerun(cropped_image, cnn, threshold = 130)
                            list_result.append(result_string)
                            if(len(result_string) <8):
                                longest_string = max(list_result, key=len)
                                len_longest_string = len(longest_string)
                                s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả kí tự. Chỉ có thể nhận diện được {len_longest_string} kí tự trên Bảng số xe</p>"
                                st.markdown(s, unsafe_allow_html=True) 
                                char = longest_string[2]
                                if ord(char) >= 65 and ord(char) <= 90:
                                    s = f"<p style='font-size:50px;'>{longest_string[:2]}-{longest_string[2:4]} {longest_string[4:]}</p>"
                                    st.markdown(s, unsafe_allow_html=True)
                                    df = pd.read_excel('./BANG_SO_XE.xlsx')
                                    data_array = df.values
                                    for i in range(len(data_array)):
                                        if np.char.strip(data_array[i][1]) == longest_string[:4]:
                                            s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                            st.markdown(s, unsafe_allow_html=True)
                                            break
                                else:
                                    s = f"<p style='font-size:50px;'>{longest_string}</p>"
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
                    list_result.append(result_string)
                    if(len(result_string) <8):
                        longest_string = max(list_result, key=len)
                        len_longest_string = len(longest_string)
                        s = f"<p style='font-size:40px;'>Không thể nhận diện tất cả kí tự. Chỉ có thể nhận diện được {len_longest_string} kí tự trên Bảng số xe</p>"
                        st.markdown(s, unsafe_allow_html=True) 
                        char = longest_string[2]
                        if ord(char) >= 65 and ord(char) <= 90:
                            s = f"<p style='font-size:50px;'>{longest_string[:2]}-{longest_string[2:4]} {longest_string[4:]}</p>"
                            st.markdown(s, unsafe_allow_html=True)
                            df = pd.read_excel('./BANG_SO_XE.xlsx')
                            data_array = df.values
                            for i in range(len(data_array)):
                                if np.char.strip(data_array[i][1]) == longest_string[:4]:
                                    s = f"<p style='font-size:40px;'>👉👈 {data_array[i][0]}</p>"
                                    st.markdown(s, unsafe_allow_html=True)
                                    break
                        else:
                            s = f"<p style='font-size:50px;'>{longest_string}</p>"
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
    DisplayDemo(st.session_state.yolov4, st.session_state.cnn, uploaded_files, version=4)
except:
    s = f"<p style='font-size:40px;'>Ảnh không thể nhận diện được</p>"
    st.markdown(s, unsafe_allow_html=True)