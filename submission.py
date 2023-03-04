# https://youtu.be/3RNPJbUHZKs
"""
Remove text from images

"""

import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
import os

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
  pred_target.save('C:/Users/Ramakrishnan/Videos/PS2/WDNet/results/'+str(i)+'.jpg')
  # if i<=20:
  #   p0.save('C:/Users/Ramakrishnan/Videos/PS2/WDNet/results/_'+str(i)+'_.jpg')
  
  
  
  

#General Approach.....
#Use keras OCR to detect text, define a mask around the text, and inpaint the
#masked regions to remove the text.
#To apply the mask we need to provide the coordinates of the starting and 
#the ending points of the line, and the thickness of the line

#The start point will be the mid-point between the top-left corner and 
#the bottom-left corner of the box. 
#the end point will be the mid-point between the top-right corner and the bottom-right corner.
#The following function does exactly that.
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

#Main function that detects text and inpaints. 
#Inputs are the image path and kreas_ocr pipeline
def inpaint_text(img_path, pipeline):
    # read the image 
    img = keras_ocr.tools.read(img_path) 
    
    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples. 
    prediction_groups = pipeline.recognize([img])
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
                 
    return(inpainted_img)

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

ids = list()
results_path = "./results/"
for file in os.listdir(results_path):
  ids.append(results_path+file)

i = 0
for img in ids:
    print(img)
    img_text_removed = inpaint_text(img, pipeline)

    plt.imshow(img_text_removed)

    cv2.imwrite('./results_2/'+str(i)+".jpg", cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))
    i+=1