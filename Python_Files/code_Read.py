import pandas as pd
import ReadabilityFeatures as read
import numpy as np

from time import time
from utility import clean,xgb_bst

start_time = time()

############### EXTRACTING AND CLEANING DATASET ###############

if flag:
    print('first time run......')
    df_train = pd.read_csv('..\data\train.csv', encoding='utf-8')
    df_train['id'] = df_train['id'].apply(str)
    
    df_test = pd.read_csv('..\data\test.csv', encoding='utf-8')
    df_test['test_id'] = df_test['test_id'].apply(str)
    
    df_all = pd.concat((df_train, df_test))
    df_all['question1'].fillna('', inplace=True)
    df_all['question2'].fillna('', inplace=True)
    flag = False        

q1_train = df_train['question1'][:1000]
q2_train = df_train['question2'][:1000]
index = df_train['is_duplicate'][:1000]

for i in range(len(q1_train)):
    if type(q1_train[i]) == float:
        q1_train[i] = 'none'
    
    elif len(q1_train[i]) <= 1:
        q1_train[i] += 'none'
    
    if type(q2_train[i]) == float:
        q2_train[i] = 'none'
    
    elif len(q2_train[i]) <= 1:
        q2_train[i] += 'none'
        
q1_train = map(clean, q1_train)
q2_train = map(clean, q2_train)
index = map(float, index)

############### EXTRACTING FEATURES ###############

CPW_Q1 = []
CPW_Q2 = []
SPW_Q1 = []
SPW_Q2 = []
LWPS_Q1 = []
LWPS_Q2 = []
LWR_Q1 = []
LWR_Q2 = []
CWPS_Q1 = []
CWPS_Q2 = []
DaleChall_Q1 = []
DaleChall_Q2 = []
ED_Dist = []
ED_Noun = []
LCS_Len = []
LCW = []

for i in range(len(q1_train)):
    CPW_Q1.append(read.CPW(q1_train[i]))
    CPW_Q2.append(read.CPW(q2_train[i]))
    SPW_Q1.append(read.SPW(q1_train[i]))
    SPW_Q2.append(read.SPW(q2_train[i]))
    LWPS_Q1.append(read.LWPS(q1_train[i]))
    LWPS_Q2.append(read.LWPS(q2_train[i]))
    LWR_Q1.append(read.LWR(q1_train[i]))
    LWR_Q2.append(read.LWR(q2_train[i]))
    CWPS_Q1.append(read.CWPS(q1_train[i]))
    CWPS_Q2.append(read.CWPS(q2_train[i]))
    DaleChall_Q1.append(read.DaleChall(q1_train[i]))
    DaleChall_Q2.append(read.DaleChall(q2_train[i]))
    ED_Dist.append(read.EditDist_Dist(q1_train[i], q2_train[i]))
    ED_Noun.append(read.EditDist_Noun(q1_train[i], q2_train[i]))
    LCS_Len.append(read.LCS_Len(q1_train[i], q2_train[i]))
    LCW.append(read.LCW(q1_train[i], q2_train[i]))
    
############### SAVING FEATURES ###############

x_train = pd.DataFrame()

x_train['CPW_Q1'] = CPW_Q1
x_train['CPW_Q2'] = CPW_Q2
x_train['SPW_Q1'] = SPW_Q1
x_train['SPW_Q2'] = SPW_Q2
x_train['LWPS_Q1'] = LWPS_Q1
x_train['LWPS_Q2'] = LWPS_Q2
x_train['LWR_Q1'] = LWR_Q1
x_train['LWR_Q2'] = LWR_Q2
x_train['CWPS_Q1'] = CWPS_Q1
x_train['CWPS_Q2'] = CWPS_Q2
x_train['DaleChall_Q1'] = DaleChall_Q1
x_train['DaleChall_Q2'] = DaleChall_Q2
x_train['ED_Dist'] = ED_Dist
x_train['ED_Noun'] = ED_Noun
x_train['LCS_Len'] = LCS_Len
x_train['LCW'] = LCW

np.save('../Extracted_Features/read', x_train)
np.save('../Extracted_Features/label', index)

############### SAVING TIME TAKEN AND XGBOOST BEST LOG LOSS ###############

end_time = time()

bst_loss = xgb_bst(x_train, index)

new_time_loss = [bst_loss, (end_time-start_time)]

try:
    time_loss = np.load('../Extracted_Features/Time_Loss_Read.npy')
except:
    time_loss = np.array([])

try:
    time_loss = np.vstack((time_loss, new_time_loss))
except:
    time_loss = list(time_loss)
    time_loss.append(new_time_loss)

np.save('../Extracted_Features/Time_Loss_Read', time_loss)