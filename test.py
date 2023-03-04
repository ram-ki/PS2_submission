from WDNet import generator
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
import os.path as osp
import os
import time
from torchvision import datasets, transforms
os.environ["CUDA_VISIBLE_DEVICES"] =  "1"

G=generator(3,3)
G.eval()
G.load_state_dict(torch.load(os.path.join('./Pretrained_WDNet/WDNet_G.pkl'), map_location=torch.device('cpu')))
# G.cuda()
root = './test_images/'
ids = list()
for file in os.listdir(root):
  #if(file[:-4]=='.jpg'):
  ids.append(root+file)
i=0
all_time=0.0
for img_id in ids:
  print(img_id)
  i+=1
  transform_norm=transforms.Compose([transforms.ToTensor()])
  img_J=Image.open(img_id)
  img_source = transform_norm(img_J)
  img_source=torch.unsqueeze(img_source,0)
  st=time.time()
  pred_target,mask,alpha,w,I_watermark=G(img_source)
  all_time+=time.time()-st
  mean_time=all_time/i
  print("mean time:%.3f"%mean_time)
  p0=torch.squeeze(img_source)
  p1=torch.squeeze(pred_target)
  p2=mask
  p3=torch.squeeze(w*mask)
  p2=torch.squeeze(torch.cat([p2,p2,p2],1))
  p0=torch.cat([p0,p1],1)
  p2=torch.cat([p2,p3],1)
  p0=torch.cat([p0,p2],2)
  p0=transforms.ToPILImage()(p0.detach().cpu()).convert('RGB')
  pred_target=transforms.ToPILImage()(p1.detach().cpu()).convert('RGB')
  pred_target.save('./results/'+str(i)+'.jpg')
  # if i<=20:
  #   p0.save('C:/Users/Ramakrishnan/Videos/PS2/WDNet/results/_'+str(i)+'_.jpg')
  
  
  
  
