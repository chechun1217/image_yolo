from ctypes import *
import random
from os import listdir
from os.path import isfile, isdir, join
#from socket import PF_CAN
import time
from queue import Queue
from collections import defaultdict
import csv
import pandas as pd
import numpy as np
import ast
import json
from PIL import Image
#import imagehash
import hashlib
import math
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing
import sys
import shlex
freewalk = [
    'freewalk_1',
]
walk_list = freewalk


def signed_encode(hash_):
    # signed compute
    hash_ = np.where(hash_ > 0, hash_, 0)
    hash_ = np.where(hash_ <= 0, hash_, 1)

    return hash_.astype(int)


def text_hash(beacon_id, RSSI):
    # if walk_list == freewalk :
    RSSI_2 = 3*RSSI*RSSI + 2*RSSI*RSSI*RSSI + 1*RSSI*RSSI*RSSI*RSSI  # freewalk
    # if walk_list == scripted_walk :
    #   RSSI_2 = 1*RSSI + 3*RSSI*RSSI*RSSI + 2*RSSI*RSSI*RSSI*RSSI       #scripted_walk
    #RSSI_2 = 3*RSSI*RSSI*RSSI
    weight = abs(RSSI_2*10)

    text_ = hashlib.blake2b(beacon_id.encode()).hexdigest()
    #text_ = hashlib.sha256(beacon_id.encode()).hexdigest()
    text_encode = np.array([])

    for x in text_:
        value_ = int(x, base=16)
        tmp = '{0:04b}'.format(value_)
        value_list = [int(char) for char in tmp]
        text_encode = np.append(text_encode, value_list)
        text_encode = text_encode.astype(int)

    # weighted
    text_encode = np.where(text_encode < 0, text_encode, text_encode*weight)
    text_encode = np.where(text_encode > 0, text_encode, -1*weight)

    return text_encode


def similarity(hash1_, hash2_):
    # print(abs(hash1_-hash2_).sum())
    #print('len of hash1:',len(hash1_))
    # return 1 - abs(hash1_-hash2_).sum()/len(hash1_)
    return 1 - (((abs(hash1_ - hash2_)).sum())/len(hash1_))


# Wireless Train Hashing
name_type = 'beacon'
wireless_path = f'./walk_data/'
Wireless_Train = pd.read_csv(join(wireless_path, 'wireless_training_noLR.csv'))
list_of_Wireless_Train_hash = []
train_label = []
for i in range(len(Wireless_Train)):
    Wireless_Train_row = Wireless_Train.iloc[i].to_dict()
    Wireless_Train_row_label = Wireless_Train_row['label']
    train_label.append(Wireless_Train_row_label)
    Wireless_Train_row.pop('label', None)
    # device1 hash encode
    Wireless_Train_hash = np.array([])
    for beacon_id, RSSI in Wireless_Train_row.items():
        hash_ = text_hash(beacon_id, RSSI)  # 每一個RSSI值被hash成一個大小512的array
        if len(Wireless_Train_hash) < 1:
            Wireless_Train_hash = hash_
        else:
            # 每個RSSI hash完的512array 8個array數值疊加起來(大小還是512)
            Wireless_Train_hash = Wireless_Train_hash + hash_

    Wireless_Train_hash = signed_encode(Wireless_Train_hash)
    list_of_Wireless_Train_hash.append(Wireless_Train_hash)


# Test Hashing 一起討論後推舉出候選人投票 寫給 image &pf
list_MDE = []
total_DE = 0
len_of_all_label = 0
K = 45

error_distri = [0]*16
hash_similarity = []
test_label = []
predict_label = []



#Wireless_Test = {'Beacon_1': 0.594202899, 'Beacon_2': 0.710144928, 'Beacon_3': 0.260869565,
#            'Beacon_4': 0.449275362, 'Beacon_5': 0.275362319,  'Beacon_7': 0.333333333, }

args=str(sys.argv[1])
Wireless_Test=args.replace("q","\"")
Wireless_Test=json.loads(Wireless_Test)

Wireless_Test_hash = np.array([])
for beacon_id, RSSI in Wireless_Test.items():
    RSSI=float(RSSI)
    hash_ = text_hash(beacon_id, RSSI)
    if len(Wireless_Test_hash) < 1:
        Wireless_Test_hash = hash_  # 因為0和512長度 不能相加
    else:
        Wireless_Test_hash = Wireless_Test_hash + hash_
Wireless_Test_hash = signed_encode(Wireless_Test_hash)
#print(Wireless_Test_hash)

# 找出 Top K 個像的
k_top_similarity = [0.0]*K
voter = [0.0]*K
for k in range(len(list_of_Wireless_Train_hash)):
    sim_ = similarity(
        list_of_Wireless_Train_hash[k], Wireless_Test_hash)
    if sim_ > (min(k_top_similarity)):
        k_top_similarity[k_top_similarity.index(
            min(k_top_similarity))] = sim_
        voter[k_top_similarity.index(
            min(k_top_similarity))] = str(int(train_label[k]))



dict={}
for key in voter:
    dict[key]=dict.get(key,0)+1

import os
import Wireless_Particle_Filter as PF

count=int(sys.argv[2])
final_position=PF.calculate(dict,count)
print(final_position)
'''
final_position_numbers = sum(c.isdigit() for c in final_position)
wireless_path = f'./walk_data/'
#用LR的set來做KNN  判斷是在這個位置的 中間 左 右

Wireless_Train_LR = pd.read_csv(join(wireless_path, 'wireless_training_LR.csv'))
list_of_Wireless_Train_LR_hash = []
train_label_LR = []
for i in range(len(Wireless_Train_LR)):
    Wireless_Train_LR_row = Wireless_Train_LR.iloc[i].to_dict()
    Wireless_Train_LR_label = Wireless_Train_LR_row['label']
    numbers = sum(c.isdigit() for c in Wireless_Train_LR_label)
    if(final_position_numbers==numbers and final_position in Wireless_Train_LR_label ):  #605的話  查看set中的 605L 605 605R
        train_label_LR.append(Wireless_Train_LR_label)
        Wireless_Train_LR_row.pop('label', None)
        # device1 hash encode
        Wireless_Train_LR_hash = np.array([])
        for beacon_id, RSSI in Wireless_Train_LR_row.items():
            hash_ = text_hash(beacon_id, RSSI)  # 每一個RSSI值被hash成一個大小512的array
            if len(Wireless_Train_LR_hash) < 1:
                Wireless_Train_LR_hash = hash_
            else:
                # 每個RSSI hash完的512array 8個array數值疊加起來(大小還是512)
                Wireless_Train_LR_hash = Wireless_Train_LR_hash + hash_

        Wireless_Train_LR_hash = signed_encode(Wireless_Train_LR_hash)
        list_of_Wireless_Train_LR_hash.append(Wireless_Train_LR_hash)

k_top_similarity = [0.0]*K
voter = [0.0]*K


for k in range(len(list_of_Wireless_Train_LR_hash)):
    sim_ = similarity(
        list_of_Wireless_Train_LR_hash[k], Wireless_Test_hash)
    if sim_ > (min(k_top_similarity)):
        k_top_similarity[k_top_similarity.index(
            min(k_top_similarity))] = sim_
        voter[k_top_similarity.index(
            min(k_top_similarity))] = train_label_LR[k]


predict_label.append(max(voter, key=voter.count))  # 票票等值
# predict_label.append(vote.index(max(vote))) # 票票不等值 similarity 為權重
print( max(voter, key=voter.count))
if(count==0):
    accuracy=[]
    accuracy.append(max(voter, key=voter.count))
    np.save('accuracy', accuracy)
else:
    accuracy = np.load('accuracy.npy')
    accuracy=np.append(accuracy,(max(voter, key=voter.count)))
    np.save('accuracy', accuracy)
    
    
if(count==5):
    result={}
    for key in accuracy:
        result[key]=result.get(key,0)+1
    print(result)
#print("final position:",final_position)
#print("predict:",final_position)
'''