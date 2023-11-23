import torch
import torchvision.transforms as transforms
import selectivesearch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import cv2

# 모델 정의
weight = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weight=weight,pretrained=True)
model.eval()

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# 이미지 로드
og_img = cv2.imread('test1.jpg')


cv2.setUseOptimized(True)
ss = cv2.Algorithm

ss.setBaseImage(og_img)
ss.switchToSelectiveSearchFast()

rects = ss.process()


# Selective Search로 후보 영역 얻기
# _, regions = selectivesearch.selective_search(og_img, min_size=2000)

# # 모델을 적용한 박스 중에 score값이 0.6 이상인 박스만 추출
boxes = []
scores = []
for rect in rects:
    x, y, w, h = rect['rect']
    # cv2.rectangle(og_img,(x,y),(x+w,y+h),(0,255,0))
    box_image = og_img[y:y+h, x:x+w]
    # cv2.imshow('test',box_image)
    # cv2.waitKey(0)
    # break

    input_tensor = preprocess_image(box_image)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    print(prediction)
    # print(np.argmax(prediction, axis=1))

    if torch.any(prediction[0]['scores'] >= 0.5):
        boxes.append(prediction[0]['boxes'].numpy())
        scores.append(prediction[0]['scores'].numpy())
        cv2.rectangle(og_img,(x,y),(x+w,y+h),(0,255,0))
        
# print(scores)


# mapping한 이미지를 cv2를 이용해 시각화
cv2.imshow('Mapped Image with NMS', og_img)
cv2.waitKey(0)
cv2.destroyAllWindows()