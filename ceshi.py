import os

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2
import glob
import torch
from PIL import Image



file_path = '/data/new.jpg'

# image = cv2.imread(file_path, cv2.IMREAD_COLOR)
# print(image.shape)
# print(image[image>0])
# cv2.imshow('image',image)
# normalize = transforms.Normalize(mean=[0.1,0.2,0.3],std=[0.1,0.2,0.15])
# image_b = F.to_tensor(image)
# print(image_b[image_b>0])
# print(image_b.size())
# #image_b = image_b[None,:]
# image_b = normalize(image_b)
#
# image_b = image_b.permute(1,2,0)
# print(image_b.size())
# image_b = image_b.numpy()
# print(type(image_b))
# cv2.imshow('image_b',image_b)
# cv2.waitKey(0)
# image = Image.open(file_path).convert('RGB')
# print(image.mode)
# print(image.size)
# image.show()
#image_rotate_45 = image.transform((200,200), Image.AFFINE, (10, 20, 30, 20, 10, 40))
#
# image_rotate_45 = image.transpose(Image.ROTATE_90)
# image_rotate_45 = image.transform((200,200),Image.QUAD,(0,0,0,500,600,500,600,0))
# print(image_rotate_45.size)
#
# image_rotate_45.show()
# def fun1(x):
#     return x*0.2
# im1_eval = Image.eval(image,fun1)
# #im1_eval.show()
# print(im1_eval.getbands())
# print(image.info)
# print(np.array(image).shape)
#
# sqe = image.getdata()
# print(len(list(sqe)))

# image = cv2.imread(file_path,cv2.IMREAD_COLOR)
# print(image.shape)
# image[:,10:20,:] = 50
# def cvt2heatmap(gray):
#     heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
#     return heatmap
# aa = cvt2heatmap(image)
# cv2.imshow('image',aa)


# class A:
#     def __init__(self):
#         self.name = 'wanghao'
#         self.value = 23
#     def change(self):
#         new_name = self.__dict__.get('name')
#         self.new_name = 'Wanghao'
#
# a = A()
# print(a.name)
# print(a.__dict__)
# a.change()
# print(a.name)
# print(a.__dict__)





images = sorted(glob.glob('/home/SENSETIME/wanghao3/图片/hongzao/look_img_3/haozao/'+"*.jpg"))
print(images)

i = cv2.imread(images[0], cv2.IMREAD_COLOR)
i_re = cv2.resize(i, (i.shape[0], i.shape[1]))

print(i.shape)
print(i_re.shape)

a = [1, 2, 3]
print(a.size)



