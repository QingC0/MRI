

import pandas as pd 
import numpy as np
import os 
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from torch.nn import MultiheadAttention,Linear, ReLU, RNN,Sigmoid,Softmax
import torch 




def NMR_matrix(matrix):
    allfr1=[]
    for i in range(matrix.shape[0]):
        matrix1=matrix[i]
        allfr2=[]
        for j in matrix1:        
            if j==0:
                allfr2.append(np.array([1,0]))
            else:
                allfr2.append(np.array([0,1]))
        allfr1.append(np.array(allfr2))
    allfr1=np.array(allfr1)
    return allfr1


def NMR_mask_matrix(matrix):
    allfr1=[]
    for i in range(matrix.shape[0]):
        matrix1=matrix[i]
        allfr2=[]
        for j in matrix1:        
            if j==0:
                allfr2.append(np.array([0,0]))
            else:
                allfr2.append(np.array([1,1]))
        allfr1.append(np.array(allfr2))
    allfr1=np.array(allfr1)
    return allfr1


def return_NMR_pad(dataframe,max_len):
    current_len=dataframe.shape[0]
    pad_len=max_len-current_len
    if pad_len>0:
        for i in range(pad_len):
            padding = [[0.0,0.0]]*dataframe.shape[1]
            dataframe=np.vstack([dataframe, padding])
    return dataframe


def return_pad(dataframe,max_len):
    current_len=dataframe.shape[0]
    pad_len=max_len-current_len
    if pad_len>0:
        for i in range(pad_len):
            padding = [0.0]*dataframe.shape[1]
            dataframe=np.vstack([dataframe, padding])
    return dataframe


# Find nonzero images for original 3D data
# return # of 2D arrays with nonzero and 4 cornors/locations of max nonzero araa
def check_max_len_img(img_frame):
    img_len=[]
    row_min, row_max, line_min, line_max = 300, -1, 300, -1
    for row in img_frame.itertuples(): #i in range(len(img_frame)):
        img_slice = np.load(row.agg_index)##img_frame.agg_index[i])
        if img_slice.shape[1] > img_slice.shape[2]:
            img_slice = np.transpose(img_slice, (0, 2, 1))
        a, b, c, d = FindCorner(img_slice)
        row_min, row_max, line_min, line_max = min(row_min,a), max(row_max,b), min(line_min,c), max(line_max,d)
        img_len.append(FindNonzero2D(img_slice))
       
    return np.max(np.array(img_len)), (row_min, row_max, line_min, line_max)


def FindCorner(m):
    # return the location/cornors of maximum nonzero 2d arrays
    # m.shape: (18/15, width, height)
    s, h, w = m.shape
    a, b, c, d = 300, -1, 300, -1
    for lg in range(s):
        r, l = np.nonzero(m[lg, :,:])
        if len(r)>0:
            a, b = min(min(r), a), max(max(r), b)
            c, d = min(min(l), c), max(max(l), d)
    return a, b, c, d

def FindNonzero2D(m):
    # Find the number of 2D arrays with nonzero
    # m.shape: (18/15, width, height) 
    s, h, w = m.shape
    l = 0
    for i in range(s):
        if np.count_nonzero(m[i,:,:]) > l:
            l += 1
    return l


def return_img_pad (dataframe,max_len, max_img_len, loc):
    # loc: the cornors/locations of the max nonzero 2d area
    second_ar=[]
    a, b, c, d = loc
    #print(b+1-a,d+1-c)
    for i in dataframe.agg_index.values:
        first_ar = []
        img=np.load(i)
        if img.shape[1] > img.shape[2]:
            img = np.transpose(img, (0, 2, 1))
        #print(img.shape)
        for j in range(img.shape[0]):
            if np.count_nonzero(img[j,:,:]) > 1:
                first_ar.append(img[j,a:b+1, c:d+1])
        first_ar = np.array(first_ar)
        
        if len(first_ar) == 0:
            first_ar = np.zeros((1,b+1-a,d+1-c))
        
        if len(first_ar) < max_img_len:
            padding = np.zeros((max_img_len-len(first_ar),first_ar.shape[1],first_ar.shape[2]))
            #print(first_ar.shape, padding.shape)
            first_ar=np.vstack([first_ar, padding])
        second_ar.append(first_ar)
    second_ar=np.array(second_ar)
    if second_ar.shape[0]<max_len:
        pad_len=max_len-second_ar.shape[0]
        padding = np.zeros((pad_len,second_ar.shape[1],second_ar.shape[2], second_ar.shape[3]))
        second_ar=np.vstack([second_ar, padding])
    return second_ar


def check_max_len(data_frame):
    max_seq=data_frame.groupby('Mouse_Name')['Days_Elapsed'].count().max()
    return max_seq

def check_max_len_img_hp(hp_path):
    #print(hp_path)
    hp_path = hp_path['agg_index']
    max_r, max_c = 0, 0
    #lon, sho = 0, 0
    for p in hp_path:
        dt = np.load(p)
        if len(dt.shape) > 2: #== (18, 192, 256):
            dt = dt.reshape(dt.shape[0], -1)
            #lon+=1
            #print('long',p)
        else:
            dt = np.swapaxes(dt, 0, 1)
            #sho+=1
            #print('shot',p)
        rs = dt.shape[0]
        if max_r < rs:
            max_r = rs
        cs = np.max(np.count_nonzero(dt, axis=1))
        if max_c < cs:
            max_c = cs
    return max_r, max_c

def return_ph_pad(dataframe, max_len, max_r, max_c):
    #print('max_len, max_r, max_c', max_len, max_r, max_c)
    first_ar = []
    for i in dataframe.agg_index.values:
        img=np.load(i)
        if len(img.shape) > 2: #== (18, 192, 256):
            img = img.reshape(img.shape[0], -1)
            img = img[~np.all(img == 0, axis=1)]
            second_ar=np.zeros((max_r, max_c))
            #print('second_ar',second_ar.shape)
            for j in range(img.shape[0]):
                k = img[j][img[j]!=0]
                second_ar[j] = np.pad(k, (0, max_c - len(k)), 'constant', constant_values=0)
            #print('long',second_ar.shape)
            img = np.asarray(second_ar)
 
        else:
            img = np.swapaxes(img, 0, 1)
            img = np.pad(img, ((0,max_r-img.shape[0]),(0,max_c - img.shape[1])), 'constant', constant_values=0)
            #print('short', img.shape)
        

        first_ar.append(img)
    first_ar = np.asarray(first_ar)  
    #print(first_ar.shape) 
    if first_ar.shape[0] < max_len:
        first_ar = np.vstack((first_ar, np.zeros((max_len - len(first_ar),max_r, max_c))))
            
     
    #print(first_ar.shape) 
    #print(kk)
    return first_ar



def LoadData(data_path, cohort = None):
    HPMRI=pd.read_csv(data_path+'all_HPMRI.csv')
    HPMRI_mask=pd.read_csv(data_path+'all_HPMRI_mask.csv')
    NMR=pd.read_csv(data_path+'all_NMR.csv')
    NMR_true=pd.read_csv(data_path+'NMR.csv')
    NMR_mask=pd.read_csv(data_path+'all_NMR_mask.csv')
    tumor=pd.read_csv(data_path+'all_tumor_volume.csv')
    tumor_mask=pd.read_csv(data_path+'all_tumor_volume_mask.csv')
    sag_path=pd.read_csv(data_path+'all_seg_path.csv')
    cor_path=pd.read_csv(data_path+'all_cor_path.csv')
    ax_path=pd.read_csv(data_path+'all_ax_path.csv')
    hp_path = pd.read_csv(data_path+'all_hp_path.csv')
    
    if cohort:
        HPMRI = HPMRI.loc[HPMRI['Cohort'] == cohort].copy()
        HPMRI_mask = HPMRI_mask.loc[HPMRI_mask['Cohort'] == cohort].copy()
        NMR = NMR.loc[NMR['Cohort'] == cohort].copy()
        NMR_true = NMR_true.loc[NMR_true['Cohort'] == cohort].copy()
        NMR_mask = NMR_mask.loc[NMR_mask['Cohort'] == cohort].copy()
        tumor = tumor.loc[tumor['Cohort'] == cohort].copy()
        tumor_mask = tumor_mask.loc[tumor_mask['Cohort'] == cohort].copy()
        sag_path = sag_path.loc[sag_path['Cohort'] == cohort].copy()
        cor_path = cor_path.loc[cor_path['Cohort'] == cohort].copy()
        ax_path = ax_path.loc[ax_path['Cohort'] == cohort].copy()
        hp_path = hp_path.loc[hp_path['Cohort'] == cohort].copy()

    return HPMRI, HPMRI_mask, NMR, NMR_true, NMR_mask, tumor, tumor_mask, sag_path, cor_path, ax_path, hp_path



class mult_attn(nn.Module):
    def __init__(self, config):
        super(mult_attn, self).__init__()
        self.MuA_1=MultiheadAttention(config['emb_size'], config['head_nu'], dropout=config['drop_rate'],bias=False) 
        self.MuA_2=MultiheadAttention(config['emb_size'], config['head_nu'], dropout=config['drop_rate'],bias=False)
        self.MuA_3=MultiheadAttention(config['emb_size'], config['head_nu'], dropout=config['drop_rate'],bias=False)
        self.relu=ReLU()
    
    
    def forward (self,emb_1,emb_2,emb_3):
        attn_output_1, _ = self.MuA_1(emb_1, emb_1, emb_1)# self_attention, output(seq,batch,emb)
        attn_output_1=self.relu(attn_output_1)
        attn_output_2,_ = self.MuA_2(attn_output_1, emb_2, emb_2) # attention_2_another
        attn_output_3,_ = self.MuA_3(attn_output_1, emb_3, emb_3) # attention_2_another
        attn_output_1=attn_output_1.permute(1, 0,2)# batch first
        attn_output_2=attn_output_2.permute(1, 0,2)# batch first
        attn_output_3=attn_output_3.permute(1, 0,2)# batch first
        attn_output_1=torch.sum(attn_output_1,dim=1)
        attn_output_2=torch.sum(attn_output_2,dim=1)
        attn_output_3=torch.sum(attn_output_3,dim=1)
        att_final=self.relu(torch.cat((attn_output_1, attn_output_2,attn_output_3),dim=1))
        return att_final
