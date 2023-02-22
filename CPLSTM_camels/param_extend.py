from pathlib import Path
from typing import Tuple, List
from sklearn.metrics import r2_score
import gcsfs
import pywt
import glob
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import csv
import sys, getopt


DATA_ROOT = r'./'
Ss = {'01022500':0.62622887793281, '01031500':0.559122720432494, '01047000':0.561180826371744, '01052500':0.615537987322707, '01054200':0.594525957550057, '01057000':0.61710021345834, '01073000':0.510061723029111}

def params(size : int = 0, P1 = [], EM1 = [], basin = '01022500'):
    #首先定义步长内各个状态参量
    """PE_t#净雨量
    EP_t#KC*EM
    P_t#降雨
    R_t#产流深
    RB_t#不透水面上产生的径流深度
    RG_t#地下径流深
    RSS_t#壤中流深度
    RS_t#地表径流深


    S = S0 #自由水蓄水深
    FS#自由水蓄水量小于等于SS的面积
    FR = FR0#产流面积相对值，即FS与Area的比值
    SSM # (1+EX)*SM，流域上自由水蓄水量最大的某点的蓄量值
    AU#与自由蓄水量S对应的自由水蓄水容量曲线的纵坐标值
    A#土壤湿度为 W时土壤含水量折算成的径流深度（ mm）
    E_t = 0.0#蒸散发
    EU_t = 0.0#上层土壤蒸散发
    EL_t = 0.0#下层土壤蒸散发
    ED_t = 0.0#深层土壤蒸散发
    WU_t = WU0
    WL_t = WL0
    WD_t = WD0
    W_t = WU_t + WL_t + WD_t

    SSM = SM * (1 + EX)
    #FR0 = 1 - pow((1 - S0/ SSM), EX)

    i #代表步长下标"""
    K = 1.0
    B = 0.3
    C = 0.15
    SM = 25
    EX = 1.5
    WUM = 15
    WLM = 75
    WM = 3361
    WDM = WM - WUM - WLM
    KKG = 0.975
    KKSS = 0.55
    steps = size
    IMP = 0.01

    E_t = 0.0#蒸散发
    EU_t = 0.0#上层土壤蒸散发
    EL_t = 0.0#下层土壤蒸散发
    ED_t = 0.0#深层土壤蒸散发
    WU_t = 0#WU0
    WL_t = 70#WL0
    WD_t = 80#WD0
    W_t = WU_t + WL_t + WD_t
    S0 = Ss[basin]
    #S0 = 33.06555771

    file1 = pd.read_csv(DATA_ROOT + f'{basin}_data.txt')
    mx = 0
    das = []
    P = P1
    EM = EM1
    if P == [] or EM == []:
        print('empty 1\n')
        for i in file1['N\tP\tE\tT\tQ']:
            da = []
            st = ''
            for j in i:
                if j == '\t':
                    da.append(float(st))
                    st = ''
                elif j == 't':
                    continue
                else:
                    st += j
            da.append(float(st))
            das.append(da)
        steps = len(das)
        for i in das:
            P.append(i[1])
            EM.append(i[2])
    print(len(P), len(EM), '\n')
    WMM = WM * (1.0 + B) / (1.0 - IMP)
    SSM = SM * (1 + EX)
    KSS= 0.5
    KG = 0.75
    #P is p
    #EM is e
    E = []
    EU = []
    EL = []
    ED = []
    W = []
    WU = []
    WL = []
    WD = []
    RG = []
    RS = []
    RSS = []
    R = []

    for i in range(steps):
    
        """蒸散发计算开始"""
        P_t = P[i] * (1 - IMP) #降在透水面的降雨量
        EP_t = K * EM[i]#降雨期间蒸发量
        if P[i] > EP_t:
            RB_t = (P[i] - EP_t) * IMP#RB是降在不透水面的降雨量
        else:
            RB_t = 0.0
        if (WU_t + P_t) >= EP_t:#上层张力蓄水量足够
        
            EU_t = EP_t
            EL_t = 0
            ED_t = 0
        
        elif (WU_t + P_t) < EP_t: #上层张力蓄水量不够
        
            EU_t = WU_t + P_t#上层蒸发量为上层蓄水量+降雨量，上层变干
            EL_t = (EP_t - EU_t) * WL_t / WLM
            if EL_t < C * (EP_t - EU_t)  and  WL_t >= C * (EP_t - EU_t): #
            
                EL_t = C * (EP_t - EU_t)
                ED_t = 0
            
            elif EL_t < C * (EP_t - EU_t)  and  WL_t < C * (EP_t - EU_t):#下层蓄量不够，触及深层
            
                EL_t = WL_t
                ED_t = C * (EP_t - EU_t) - EL_t
            
        
        E_t = EU_t + EL_t + ED_t#已拥有，符号为E
        PE_t = P_t - E_t#P_T已拥有，符号为P
        """蒸散发计算结束"""

        """子流域产流量计算开始"""
        if PE_t <= 0: #净雨小于0，降雨全部蒸发
        
            R_t = 0.0#不产流
            W_t = W_t + PE_t#更新含水量
        
        else:
        

            A = WMM * (1 - pow((1.0 - W_t / WM), 1.0 / (1 + B)))
            # 土壤湿度折算净雨量 +降水后蒸发剩余雨量 <流域内最大含水容量
            if (A + PE_t) < WMM:
            
                R_t = PE_t + W_t + WM * pow((1 - (PE_t + A) / WMM), (1 + B)) - WM + RB_t
            
            # 土壤湿度折算净雨量 +降水后蒸发剩余雨量 >流域内最大含水容量
            else:
            
                # 流域内的产流深度计算
                R_t = PE_t + W_t - WM + RB_t
            
        
        #三层蓄水量的计算：WU，WL，WD
        if WU_t + P_t - EU_t - R_t <= WUM:#表层未达到蓄水容量
        
            WU_t = WU_t + P_t - EU_t - R_t
            WL_t = WL_t - EL_t
            WD_t = WD_t - ED_t
        
        else:
        
            WU_t = WUM#表层达到蓄水容量
            if WL_t - EL_t + (WU_t + P_t - EU_t - R_t - WUM) < WLM:#下层未达到蓄水容量
            
                WL_t = WL_t - EL_t + (WU_t + P_t - EU_t - R_t - WUM)
                WD_t = WD_t - ED_t
            
            else:#下层达到蓄水容量
            
                WL_t = WLM
                if (WD_t - ED_t + WL_t - EL_t + (WU_t + P_t - EU_t - R_t - WUM) - WLM) <= WDM:#深层未达到蓄水容量
                    WD_t = WD_t - ED_t + WL_t - EL_t + (WU_t + P_t - EU_t - R_t - WUM) - WLM
                else:
                    WD_t = WDM
            
        
        """子流域产流量计算结束"""

        """三水源划分汇流计算"""
        if PE_t > 0:#如果净雨大于0
        
            FR = (R_t - RB_t) / PE_t
            S = S0
            AU = SSM * (1 - pow((1 - S / SM), 1 / (1 + EX)))
            if PE_t + AU < SSM:
            
                RS_t = FR * (PE_t + S - SM + SM * pow((1 - (PE_t + AU) / SSM), EX + 1))
                RSS_t = FR * KSS * (SM - SM * pow((1 - (PE_t + AU) / SSM), EX + 1))
                RG_t = FR * KG * (SM - SM * pow((1 - (PE_t + AU) / SSM), EX + 1))
                S0 = (1 - KSS - KG) * (SM - SM * pow((1 - (PE_t + AU) / SSM), EX + 1))
            
            elif PE_t + AU >= SSM:
            
                RS_t = FR * (PE_t + S - SM)
                RSS_t = SM * KSS *FR
                RG_t = SM * KG * FR
                S0 = (1 - KSS - KG) * SM
            
            RS_t += RB_t
            R_t = RS_t + RSS_t + RG_t

            FR0 = FR
        
        elif PE_t <= 0:
        
            S = S0
            FR = (1 - pow((1 - W_t / WM), B / (1 + B)))
            #RSS_t = 0.0
            #RG_t = 0.0
            RSS_t = S * KSS * FR
            RG_t = S * KG * FR
            RS_t = RB_t
            R_t = RS_t + RSS_t + RG_t

            S0 = S * (1 - KSS - KG)
            FR0 = FR
        
        """三水源划分汇流计算结束"""
        #状态量保存
        E.append(E_t)
        EU.append(EU_t)
        EL.append(EL_t)
        ED.append(ED_t)
        W.append(W_t)
        WU.append(WU_t)
        WL.append(WL_t)
        WD.append(WD_t)
        RG.append(RG_t)
        RS.append(RS_t)
        RSS.append(RSS_t)
        R.append(R_t)
    return E, EU ,EL, ED, W, WU, WL, WD, RG, RS, RSS, R


def m(basin):
    E, EU ,EL, ED, W, WU, WL, WD, RG, RS, RSS, R = params(P1 = [], EM1 = [], basin = basin)
    print(len(E), '\n')
    

    """f = open(DATA_PATH + '00000001_help.csv','w',encoding='utf-8')
    csv_w = csv.writer(f)
    csv_w.writerow(["EE","WW", "RG", "R", "RS", "RSS", "EU","EL", "ED", "WU", "WL", "WD"])
    for i in range(len(E)):
        csv_w.writerow([E[i],W[i], RG[i], R[i], RS[i], RSS[i], EU[i], EL[i], ED[i], WU[i], WL[i], WD[i]])
    f.close()"""

    datas = ["EE\tWW\tRG\tR\tRS\tRSS\tEU\tEL\tED\tWU\tWL\tWD"]
    for i in range(len(E)):
        da = [str(E[i]), str(W[i]), str(RG[i]), str(R[i]), str(RS[i]), str(RSS[i]), str(EU[i]), str(EL[i]), str(ED[i]), str(WU[i]), str(WL[i]), str(WD[i])]
        datas.append('\t'.join(da))
    tx = '\n'.join(datas)
    with open(DATA_ROOT + f'{basin}_help.txt',"w") as f:
        f.write(tx)

if __name__ == '__main__':
    basin = '01022500'
    ARGS = sys.argv[1:]
    if len(ARGS) > 0:
        opts, args = getopt.getopt(ARGS, 'hb:', ['help', 'basin='])
        for i, val in opts:
            if i in ['-b', '--basin']:
                basin = val 

    m(basin)
