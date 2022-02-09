import math
import numpy as np
import cv2


# poses.txt의 한 줄 구조는 r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
all = set(i for i in range(12))
T_idx = {3,7,11}
R_idx = all - T_idx

def T_RMSE(ground_truth, our):

    SUM = 0
    
    for g,o in zip(ground_truth, our):
        g_t = [g[i] for i in list(T_idx)]
        o_t = [o[i] for i in list(T_idx)]
        SUM += np.sqrt(np.sum((np.array(g_t) - np.array(o_t))**2)/3)
    
    return SUM


def R_RMSE(ground_truth, our):

    SUM = 0
    
    for g,o in zip(ground_truth, our):
        g_t = [g[i] for i in list(R_idx)]
        o_t = [o[i] for i in list(R_idx)]
        SUM += np.sqrt(np.sum((np.array(g_t) - np.array(o_t))**2)/9)
    
    return SUM

def all_RMSE(ground_truth, our):

    SUM = 0    
    for g,o in zip(ground_truth, our):
        SUM += np.sqrt(np.sum((np.array(g) - np.array(o))**2)/len(g))
    
    return SUM


if __name__ == "__main__":
    ground_truth = "../ORB_SLAM2/dataset/poses/00.txt"
    our = "../ORB_SLAM2/seq00_stereo/CameraTrajectory.txt"

    
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
    
    print(len(gt), len(ours))
  
    t_rmse = T_RMSE(gt, ours)
    r_rmse = R_RMSE(gt, ours)
    print(f"T_RMSE : {t_rmse:.6f}")
    print(f"R_RMSE : {r_rmse:.6f}")
    