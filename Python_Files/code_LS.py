import pandas as pd
import LexAndSemFeatures as LS
import numpy as np

from time import time
from utility import clean, xgb_bst

start_time = time()

############### EXTRACTING AND CLEANING DATASET ###############

if flag:
    print('first time run......')
    df_train = pd.read_csv('../data/train.csv', encoding='utf-8')
    df_train['id'] = df_train['id'].apply(str)
    
    df_test = pd.read_csv('../data/test.csv', encoding='utf-8')
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

############### TRAINING MODELS ###############

lst = []
lst.extend(q1_train)
lst.extend(q2_train)

bow_model = LS.train_BOW(lst)
bigram_model = LS.train_bigram(lst)
trigram_model = LS.train_trigram(lst)
W2V_model = LS.W2V_train(q1_train, q2_train)
LDA_model = LS.LDA_train(lst)

############### EXTRACTING FEATURES ###############

Len_Q1 = []
Len_Q2 = []
SubString = []
BOW_Q1 = []
BOW_Q2 = []
bigram_Q1 = []
bigram_Q2 = []
trigram_Q1 = []
trigram_Q2 = []
W2V_Vec = []
LDA = []

for i in range(len(q1_train)):
    Len_Q1.append(LS.length(q1_train[i]))
    Len_Q2.append(LS.length(q2_train[i]))
    SubString.append(LS.substringCheck(q1_train[i], q2_train[i]))
    BOW_Q1.append(LS.Sum_BOW(q1_train[i], bow_model))
    BOW_Q2.append(LS.Sum_BOW(q2_train[i], bow_model))
    bigram_Q1.append(LS.sum_bigram(q1_train[i], bigram_model))
    bigram_Q2.append(LS.sum_bigram(q2_train[i], bigram_model))
    trigram_Q1.append(LS.sum_trigram(q1_train[i], trigram_model))
    trigram_Q2.append(LS.sum_trigram(q2_train[i], trigram_model))
    W2V_Vec.append(LS.W2V_Vec(q1_train[i], q2_train[i], W2V_model))
    LDA.append(LS.LDA(q1_train[i], q2_train[i], LDA_model))
    
############### SAVING FEATURES ###############

x_train = pd.DataFrame()

x_train['Length_Q1'] = Len_Q1
x_train['Length_Q2'] = Len_Q2
x_train['SubStringCheck'] = SubString
x_train['BOW_Q1'] = BOW_Q1
x_train['BOW_Q2'] = BOW_Q2
x_train['Bigram_Q1'] = bigram_Q1
x_train['Bigram_Q2'] = bigram_Q2
x_train['Trigram_Q1'] = trigram_Q1
x_train['Trigram_Q2'] = trigram_Q2
x_train['W2V_Vec'] = W2V_Vec
x_train['LDA'] = LDA

np.save('../Extracted_Features/lex', x_train)
np.save('../Extracted_Features/label', index)

############### SAVING TIME TAKEN AND XGBOOST BEST LOG LOSS ###############

end_time = time()

bst_loss = xgb_bst(x_train, index)

new_time_loss = [bst_loss, (end_time-start_time)]

try:
    time_loss = np.load('../Extracted_Features/Time_Loss_LS.npy')
except:
    time_loss = np.array([])

try:
    time_loss = np.vstack((time_loss, new_time_loss))
except:
    time_loss = list(time_loss)
    time_loss.append(new_time_loss)

np.save('../Extracted_Features/Time_Loss_LS', time_loss)