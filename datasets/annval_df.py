import json
import cv2
import pandas as pd
import numpy as np
from pycocotools.coco import COCO


with open('instances_val2017.json', 'r') as file:
    data1 = json.load(file)

images = data1['images']
categories = data1['categories']
annotations1 = data1['annotations']

# print(annotations)
# print(categories)
# print(len(annotations))

ann_val = pd.DataFrame(annotations1)
print(ann_val)
annval = ann_val.columns[3][36778]
print(annval)



