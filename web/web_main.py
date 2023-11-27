import sys
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import random

import importlib

colors = []
for _ in range(50):
    color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    colors.append(color)

def get_image(image_path):
    global image
    image = st.image(image_path,use_column_width=True)

def get_label_list():
    label_list=['background']

    with open("datasets/coco-labels-2014_2017.txt", "r") as f:
        for line in f:
            label_list.append(line.strip('\n'))
    
    return label_list

st.set_page_config(layout="wide",page_title='원샷원픽',page_icon='web/asssets/osop_icon.png')

# not_image=Image('web/asssets/not_found_img.png')
IMAGE_PATH="datasets/images/val2017/000000356094.jpg"
org_image=None
pick_image=None
select_oj_list=[]
ann_list=[
    {'segmentation': [[185.32, 381.13, 190.57, 336.55, 221.16, 270.99, 218.54, 201.93, 222.04, 157.35, 194.94, 158.22, 191.44, 135.49, 202.8, 109.27, 232.53, 92.66, 222.04, 82.17, 207.18, 79.55, 226.41, 67.31, 250.88, 35.84, 284.98, 39.34, 305.96, 58.57, 306.83, 90.91, 291.09, 97.91, 296.34, 111.02, 328.68, 137.24, 322.56, 173.96, 314.7, 210.67, 319.94, 247.39, 329.56, 294.59, 342.67, 384.63, 289.35, 382.88, 265.74, 370.64, 243.89, 384.63]], 
     'area': 37444.61005, 
     'iscrowd': 0, 
     'image_id': 356094, 
     'bbox': [185.32, 35.84, 157.35, 348.79], 
     'category_id': 1, 
     'id': 451122}, 
    {'segmentation': [[420.8, 117.53, 424.88, 129.79, 427.61, 133.87, 423.52, 140.69, 420.8, 144.77, 419.44, 153.63, 414.67, 163.84, 420.8, 172.02, 420.12, 190.41, 416.03, 207.43, 413.99, 219.01, 409.22, 226.51, 405.81, 246.94, 423.52, 246.94, 422.84, 234.0, 419.44, 221.06, 422.84, 208.12, 425.57, 195.86, 435.1, 203.35, 433.74, 211.52, 430.33, 229.23, 426.93, 246.94, 425.57, 256.47, 431.7, 281.68, 435.78, 309.6, 437.83, 319.82, 425.57, 328.67, 436.46, 332.08, 451.45, 324.59, 457.58, 326.63, 458.94, 327.99, 449.4, 331.4, 450.09, 335.48, 470.52, 332.76, 468.48, 315.05, 464.39, 298.7, 463.71, 272.82, 462.35, 262.6, 464.39, 240.81, 471.2, 225.82, 469.84, 208.8, 469.84, 190.41, 469.16, 161.8, 465.75, 148.86, 457.58, 133.87, 449.4, 129.11, 445.32, 120.93, 422.16, 119.57]], 
     'area': 8569.869399999996, 
     'iscrowd': 0, 
     'image_id': 356094, 
     'bbox': [405.81, 117.53, 65.39, 217.95], 
     'category_id': 1, 
     'id': 459049}, 
    {'segmentation': [[306.03, 84.7, 387.76, 97.44, 430.21, 102.75, 455.69, 109.11, 459.4, 120.26, 455.15, 124.5, 304.44, 91.6, 306.03, 85.23]], 
     'area': 1846.9890499999997, 
     'iscrowd': 0, 
     'image_id': 356094, 
     'bbox': [304.44, 84.7, 154.96, 39.8], 
     'category_id': 39, 
     'id': 628097}, 
    {'segmentation': [[421.76, 232.49, 423.94, 241.19, 424.12, 245.53, 421.4, 249.7, 413.98, 253.86, 408.91, 253.86, 404.92, 248.97, 402.75, 233.4, 404.38, 228.87, 409.63, 228.87, 412.17, 228.87, 419.95, 231.77]], 
     'area': 423.4996499999997, 
     'iscrowd': 0, 
     'image_id': 356094, 
     'bbox': [402.75, 228.87, 21.37, 24.99], 
     'category_id': 40, 
     'id': 1472121}
]




### categories ###
label_list = get_label_list()
# print(label_list)

categories=['전체']
if len(label_list)>1:
    for ann in ann_list:
        if  label_list[int(ann['category_id'])] not in categories:
            categories.append(label_list[int(ann['category_id'])])

### object ###
def select_class_label(label):
    global label_list, ann_list
    oj_dict = {}
    if label == '전체':
        for idx, ann in enumerate(ann_list):
            oj_dict[str(idx+1)+"번"]=ann
    else:
        for idx, ann in enumerate(ann_list):
            if label_list.index(label) == ann['category_id']:
                oj_dict[str(idx+1)+"번"]=ann

    return oj_dict

### masking img ###
def masking_img(datas):
    global colors

    file_byte=np.array(bytearray(image.read()),dtype=np.uint8)
    img=cv2.imdecode(file_byte,1)
    mask=None

    for obj in datas:
        segmentation = obj['segmentation']

        color=colors[random.randint(0,49)]

        if isinstance(segmentation, list):
            h, w = img.shape[:2]

            mask = np.zeros((h,w), dtype=np.uint8)

            for seg in segmentation:
                poly = np.array(seg).reshape((-1,2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 255)
            
            mask_color = np.stack([mask]*3, axis=-1)
            for i in range(3):
                mask_color[..., i][mask_color[..., i] == 255] = color[i]

            alpha = ((mask_color >0).max(axis=2)*128).astype(np.uint8)
            rgba_mask = np.concatenate([mask_color, alpha[:,:, np.newaxis]], axis=2)
            image_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            # print(image_rgba.shape, rgba_mask.shape)
            image_rgba = cv2.addWeighted(image_rgba, 1, rgba_mask, 0.5, 0)

            mask=cv2.cvtColor(rgba_mask,cv2.COLOR_RGBA2BGR)

            img = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)

    return img, mask

        

### object remove & lama painting ###
def remove_lama():
    pass



st.title("ONE SHOT ONE PICK", anchor="/")

empty1, con1, empty2 = st.columns([1,8,1])

datas=[]


with empty1:
    st.empty()

with con1:
    st.session_state['image']=st.file_uploader('IMAGE',type=['jpg','jpeg','png'])
    image=st.session_state['image']
    tab1, tab2 = st.tabs(['원샷', '원픽'])
    with tab1:
        col1, col2 = st.columns([7,3])
        with col1:
            if image:
                    st.image(image,use_column_width=True)
            else:
                st.image('web/asssets/not_found_img.png', use_column_width=True)

        with col2:
            if image:
                category_id=st.selectbox('제거선택',categories)
                obj_dict=select_class_label(category_id)
                select_oj_list=st.multiselect(category_id,options=obj_dict.keys())
                # print(select_oj_list)

                for key, val in obj_dict.items():
                    if key in select_oj_list:
                        datas.append(val)

                if len(datas)>0:
                    mask_img, mask = masking_img(datas)
                    # st.image(mask,channels='BGR',use_column_width=True)
                    print(mask.shape)
                    st.image(mask_img,channels='BGR',use_column_width=True)

                st.button('제거하기',use_container_width=True,on_click=remove_lama())
            else:
                st.warning('먼저 이미지를 넣어주세요 ⬆︎⬆︎⬆︎')


    with tab2:
        col1, col2 = st.columns([7,3])
        with col1:
            if pick_image:
                st.image(pick_image,use_column_width=True)
            else:
                st.image('web/asssets/not_found_img.png',use_column_width=True)

        with col2:
            if pick_image:
                st.download_button('다운로드', pick_image)
            else:
                st.warning('다운받을 이미지가 없습니다!')
    
with empty2:
    st.empty()

