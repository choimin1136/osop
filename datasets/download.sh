cd datasets/images

wget http://images.cocodataset.org/zips/train2017.zip -O coco_train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O coco_val2017.zip

unzip coco_train2017.zip
unzip coco_val2017.zip

rm coco_train2017.zip
rm coco_val2017.zip

cd ..

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O coco_ann2017.zip

unzip coco_ann2017.zip
rm coco_ann2017.zip