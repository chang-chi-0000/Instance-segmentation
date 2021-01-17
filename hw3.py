# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 10:55:06 2021

@author: 張濟
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import glob
import os
import torch
import torchvision
from IPython.display import display 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data.dataset import Dataset
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test", type=int)
args = parser.parse_args()
# SYSTEM_ROOT = '/Volumes/Macintosh HD/Users/darklanx'
SYSTEM_ROOT = ''
ROOT = './'
sys.path.append(ROOT)

import random
import torch

from torchvision.transforms import functional as F
from utils import binary_mask_to_rle
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_submission(json, dir, model):
    cocoGt = COCO(json)
    coco_dt = []
    with torch.no_grad():
        model.eval()
        for imgid in cocoGt.imgs:
            print(imgid)
            img_path = os.path.join(dir, cocoGt.loadImgs(ids=imgid)[0]['file_name'])
            image = Image.open(img_path).convert("RGB")
            image = torchvision.transforms.functional.to_tensor(image.copy())
            image = image.to(device)
            prediction = model([image])[0]
            masks = prediction['masks'].cpu()
            masks.squeeze_(1)
            categories = prediction['labels'].cpu()  
            scores = prediction['scores'].cpu()
            n_instances = len(scores)    
            if len(categories) > 0: # If any objects are detected in this image
                for i in range(n_instances): # Loop all instances
                    # save information of the instance in a dictionary then append on coco_dt list
                    pred = {}
                    pred['image_id'] = imgid # this imgid must be same as the key of test.json
                    pred['category_id'] = int(categories[i])
                    mask = masks[i, :, :].numpy()
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0 
                    pred['segmentation'] = binary_mask_to_rle(mask) # save binary mask to RLE, e.g. 512x512 -> rle
                    pred['score'] = float(scores[i])
                    coco_dt.append(pred)
    return coco_dt

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

DATA_ROOT = os.path.join(ROOT, "dataset")

TRAIN_ANNO = os.path.join(DATA_ROOT, "pascal_train.json")
TEST_ANNO = os.path.join(DATA_ROOT, "test.json")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")

coco = COCO(TRAIN_ANNO) # load training annotations
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
num_cats = len(coco.cats) + 1

imgIds = 5 # Use the key above to retrieve information of the image
img_info = coco.loadImgs(ids=imgIds)
print(img_info)
img_path = os.path.join(TRAIN_DIR,img_info[0]['file_name'] )
image = cv2.imread(img_path)[:,:,::-1] # In your implementation, you should find this image in **train_images/** folders
plt.imshow(image)
# Use the imgIds to find all instance ids of the image
annids = coco.getAnnIds(imgIds=imgIds)
anns = coco.loadAnns(annids)
print("Number of instances: ", len(annids))
instance_id = 0
print(anns[instance_id].keys()) # check the information of the first instance of the image

class TrainingDataset(Dataset):
    def __init__(self, imgDir, annotation, transforms=None):
        self.imgDir = imgDir
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgDir = imgDir
        self.annotation = annotation
        self.coco = COCO(annotation)
    def __getitem__(self, idx):
        img_id = list(self.coco.imgs.keys())[idx]
        img_info = self.coco.loadImgs(ids=img_id)
        img_path = os.path.join(self.imgDir, img_info[0]['file_name'])
        img = Image.open(img_path).convert("RGB")
        annids = self.coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annids)
        masks = []
        boxes = []
        labels = []

        for i, ann in enumerate(anns):
            mask = coco.annToMask(ann)
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            masks.append(mask)
            labels.append(ann['category_id'])
    
        target = {}
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        target['masks'] = masks
        target['boxes'] = boxes
        target['labels'] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"]  = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(anns),), dtype=torch.int64)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    def __len__(self):
        return len(self.coco.imgs)

transforms = get_transform(True)
training_dataset = TrainingDataset(TRAIN_DIR, TRAIN_ANNO, transforms=transforms)
testing_dataset = TrainingDataset(TRAIN_DIR, TRAIN_ANNO, transforms=get_transform(False))
print(len(training_dataset))
training_dataset[0]

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_cats, pretrained_backbone=True)
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_cats)
# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                hidden_layer,
                                                num_cats)


model.to(device)

from engine import train_one_epoch, evaluate
import utils

def collate_fn(batch):
    return tuple(zip(*batch))
data_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
data_loader_test = torch.utils.data.DataLoader(
        testing_dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
#a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

def plot():
    y = []
    with open("training_loss.txt", 'r') as f:
        line = f.readline()
        while line:
            y.append(line)
            line = f.readline()
    plt.figure()
    plt.plot(y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
if args.test is None:
    LOAD = 30
    if LOAD != 0:
        model.load_state_dict(torch.load("./logs/{}.pth".format(LOAD)))
    num_epochs = 0
    for epoch in range(LOAD+1, LOAD+1+num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), "logs/{}.pth".format(epoch))
    plot()
else:
    with torch.no_grad():
        
        LOAD = args.test
        model.load_state_dict(torch.load("./logs/{}.pth".format(LOAD)))
        
        # submission_fname = "submission_train.json"
        submission_fname = "0650726.json"
        if not os.path.exists(os.path.join(ROOT, submission_fname)):
            print("Creating submission file...")
            # coco_dt = create_submission(TRAIN_ANNO, TRAIN_DIR, model)
            coco_dt = create_submission(TEST_ANNO, TEST_DIR, model)
            with open(submission_fname, "w") as f:
                json.dump(coco_dt, f)
            print("submission saved!")
        '''
        cocoGt = COCO(os.path.join(DATA_ROOT ,"pascal_train.json"))
        cocoDt = cocoGt.loadRes(os.path.join(ROOT, submission_fname))

        cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        '''
        
        '''
        model.eval()
        img_path = "/home/hesu215/hw3/dataset/train_images/2007_000033.jpg"
        img = Image.open(img_path).convert("RGB")
        # image = cv2.imread(img_path)[:,:,::-1] # load image
        plt.subplots(nrows=2, ncols=5, figsize=(45, 12))
        plt.subplot(2, 5, 1)
        plt.imshow(img)
        x = torchvision.transforms.functional.to_tensor(img)
        x = x.to(device)
        predictions = model([x])           # Returns predictions

        
        for i, mask in enumerate(predictions[0]['masks'].cpu()[0:9]):
            mask.squeeze_(0)
            mask[mask<0.5] = 0
            mask[mask>=0.5] = 1
            print(mask.shape)
            plt.subplot(2, 5, i+1+1)
            id = predictions[0]['labels'][i].item()
            plt.title("Instance {}, category={}".format(i+1, coco.cats[id]['name']))
            plt.imshow(mask)
        plt.savefig("fig.jpg")
        ''' 