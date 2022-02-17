import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

from RMSE import T_RMSE
from RMSE import R_RMSE

def visual_odometry(ours, gt):
    T_idx = [3,7,11]
    traj = np.zeros((1000,2000),dtype=np.uint8)
    traj = cv2.cvtColor(traj,cv2.COLOR_GRAY2BGR)
    img = traj.copy()
    textOrg1 = (10,30)
    textOrg2 = (10,80)
    textOrg3 = (10,100)

    for i in range(len(ours)):
        x = int(ours[i][3]) + 1000
        y = int(ours[i][11]) + 100

        gt_x = int(gt[i][3]) + 1000
        gt_y = int(gt[i][11]) + 100

        cv2.circle(img, (x,y), 1 , (0,0,255), 2)
        cv2.circle(img, (gt_x, gt_y), 1 , (0,255,0), 2)


    t_rmse = T_RMSE(gt, ours)
    r_rmse = R_RMSE(gt, ours)
    
    text = f"Translation Error: {t_rmse:.6f}, Rotation Error: {r_rmse:.6f}"  
    cv2.putText(img, text, textOrg1, cv2.FONT_HERSHEY_PLAIN,1, (255,255),1,8)
    
    cv2.putText(img, "Ground Truth: ", textOrg2, cv2.FONT_HERSHEY_PLAIN,1, (255,255),1,8)
    cv2.line(img, (135,75), (165,75), (0,255,0), 2)
    cv2.putText(img, "Ours: ", textOrg3, cv2.FONT_HERSHEY_PLAIN,1, (255,255),1,8)
    cv2.line(img, (135,95), (165,95), (0,0,255), 2)

    cv2.imshow("trajectory", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    ground_truth = "../ORB_SLAM2/dataset/poses/00.txt"
    our = "../ORB_SLAM2/seq00_stereo/CameraTrajectory.txt"

    #ground_truth = sys.argv[1]
    #our = sys.argv[2]
    
    with open(ground_truth, 'r') as f:
        gt = []         
        while True:
            arr = f.readline().rstrip()
            if arr:
                l = list(map(float, arr.split(' ')))
                gt.append(l)
            else:
                break
    
    with open(our, 'r') as f:
        ours = []         
        while True:
            arr = f.readline().rstrip()
            if arr:
                l = list(map(float, arr.split(' ')))
                ours.append(l)
            else:
                break

    print(max(i[3] for i in ours))
    visual_odometry(ours, gt)
    