# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:04:47 2020

@author: 91742
"""
import numpy as np
import cv2
from darkflow.net.build import TFNet
options = {"model": "cfg/tiny-yolo-voc-1c.cfg",
           "threshold":0.1,
           "load": -1,
           "gpu": 0.6}
tfnet2 = TFNet(options)
tfnet2.load_from_ckpt()
def plot_box(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence'] * 100
        label = result['label'] + " " + str(round(confidence, 2))
        
        newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
        newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_ITALIC , 1, (0, 255, 0), 2)
        
    return newImage
cap = cv2.VideoCapture(r"C:\Users\91742\Desktop\minip\darkflow-master\auto.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 


while(True):
    ret, frame = cap.read()
    
    if ret == True:
        frame = np.asarray(frame)      
        results = tfnet2.return_predict(frame)  
        new_frame = plot_box(frame, results)
        cv2.imshow('frame', new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()