import sys
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import random
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import mask_rcnn
from torchvision.transforms import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from simple_lama_inpainting import SimpleLama

import matplotlib.pyplot as plt

import importlib

print(len(mask_rcnn._COCO_CATEGORIES))
class_names=mask_rcnn._COCO_CATEGORIES
model = maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()
# model.cuda()

colors = []
for _ in range(50):
    color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    colors.append(color)

# def get_image(image_path):
#     global image
#     image = st.image(image_path,use_column_width=True)

# def get_label_list():
#     label_list=['background']

#     with open("datasets/coco-labels-2014_2017.txt", "r") as f:
#         for line in f:
#             label_list.append(line.strip('\n'))
    
#     return label_list

st.set_page_config(layout="wide",page_title='원샷원픽',page_icon='web/assets/osop_icon.png')

# not_image=Image('web/asssets/not_found_img.png')
# IMAGE_PATH="datasets/images/val2017/000000356094.jpg"
org_image=None
pick_image=None
select_oj_list=[]

### categories ###
# label_list = get_label_list()
# print(label_list)

categories=['전체']
# if len(label_list)>1:
#     for ann in ann_list:
#         if  label_list[int(ann['category_id'])] not in categories:
#             categories.append(label_list[int(ann['category_id'])])

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
def masking_img(img, datas):
    global colors
    mask=None
    masks=[]

    for mask in datas:

        color=colors[random.randint(0,49)]

        if isinstance(mask, np.ndarray):
            _,mask_color=cv2.threshold(mask,0.5,255,cv2.THRESH_BINARY)
            
            # mask_color=cv2.imdecode()
            for i in range(3):
                mask_color[..., i][mask_color[..., i] == 255] = color[i]

            alpha = ((mask_color >0).max(axis=2)*128).astype(np.uint8)
            rgba_mask = np.concatenate([mask_color, alpha[:,:, np.newaxis]], axis=2)
            image_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            image_rgba = cv2.addWeighted(image_rgba, 1, rgba_mask, 0.7, 0, dtype = cv2.CV_8U)

            masks.append(cv2.cvtColor(rgba_mask,cv2.COLOR_RGBA2BGR))

            img = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)
    c_mask=np.zeros(img.shape[:2],dtype=np.uint8)
    c_mask=np.stack([c_mask]*3, axis=-1)
    for m in masks:
        c_mask=np.add(c_mask,m,dtype=np.float32)
    c_mask=cv2.cvtColor(c_mask, cv2.COLOR_BGR2GRAY)
    _,c_mask=cv2.threshold(c_mask,0.5,255,cv2.THRESH_BINARY)

    return img, c_mask

def load_mask_rcnn(image):
    masks={}
    file_byte=np.array(bytearray(image.read()),dtype=np.uint8)
    # print(type(file_byte))
    
    img=cv2.imdecode(file_byte,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # st.image(img)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    img_tensor = preprocess(img)
    # img_tensor = img_tensor.unsqueeze(0).cuda()
    img_tensor = img_tensor.unsqueeze(0)
    
    # print(img_tensor.shape)
    
    predict = model(img_tensor)
    # print(predict)
    predict=predict[0]

    for i in range(len(predict['labels'])):
        if predict['scores'].data[i].item() > 0.8:
            img_mask = (predict['masks'][i].squeeze(0).detach().cpu().numpy())
            
            img_mask=np.stack([img_mask]*3, axis=-1)
            masks[f'{i+1}번']=img_mask

            # print(img.shape)
            if mask_rcnn._COCO_CATEGORIES[predict['labels'][i]] not in categories:
                categories.append(mask_rcnn._COCO_CATEGORIES[predict['labels'][i]])
            # show mask image
            #st.image(img)
    return img, masks

def mask_download_image(image):
    """numpy 배열의 이미지를 다운로드합니다."""
    succ, enc_image = cv2.imencode('.jpg', image)
    image_bytes = enc_image.tobytes()
    
    filename = "image_mask.jpg"
    mime_type = "image/jpg"
    st.download_button(
        label="이미지 다운로드",
        data=image_bytes,
        file_name=filename,
        mime=mime_type,
    )
def osop_download_image(image):
    """numpy 배열의 이미지를 다운로드합니다."""
    succ, enc_image = cv2.imencode('.jpg', image)
    image_bytes = enc_image.tobytes()
    
    filename = "osop_image.jpg"
    mime_type = "image/jpg"
    st.download_button(
        label="이미지 다운로드",
        data=image_bytes,
        file_name=filename,
        mime=mime_type,
    )

 

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
                    img, masks=load_mask_rcnn(image=image)
            else:
                st.image('web/assets/not_found_img.png', use_column_width=True)

        with col2:
            if image:
                if len(categories):
                    # 카테고리 분류 및 선택
                    category_id=st.selectbox('제거선택',categories)
                    # obj_dict=select_class_label(category_id)
                    select_oj_list=st.multiselect(category_id,options=masks.keys())
                # print(select_oj_list)
                else:
                    st.warning('객체를 검출하지 못했습니다.')

                for key, val in masks.items():
                    if key in select_oj_list:
                        datas.append(val)

                # 미리보기 & 제거
                if len(datas)>0:
                    # print(datas)
                    mask_img, mask = masking_img(img,datas)
                    st.image(np.stack([mask]*3, axis=-1),channels='BGR',clamp=True,use_column_width=True)
                    mask_download_image(mask)
                        
                    # print(mask.shape)
                    st.image(mask_img,channels='RGB',use_column_width=True)

                    if st.button('제거하기',use_container_width=True):
                        remove_lama()
                else:
                    st.info('제거 요소를 선택하세요.', icon="ℹ️")
            else:
                st.warning('먼저 이미지를 넣어주세요 ⬆︎⬆︎⬆︎')


    with tab2:
        col1, col2 = st.columns([7,3])
        with col1:
            if pick_image:
                st.image(pick_image,use_column_width=True)
            else:
                st.image('web/assets/not_found_img.png',use_column_width=True)

        with col2:
            if pick_image:
                osop_download_image(pick_image)
            else:
                st.warning('다운받을 이미지가 없습니다!')
    
with empty2:
    st.empty()

