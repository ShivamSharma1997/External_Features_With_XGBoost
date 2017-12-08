import gensim
import numpy as np
import pandas as pd

from time import time
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from scipy.stats import skew, kurtosis
from utility import clean, tokenize, xgb_bst
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

start_time = time()

stop_words = stopwords.words('english')

############### EXTRACTING AND CLEANING DATASET ###############
if flag:
    print('first time run......')
    df_train = pd.read_csv('train.csv', encoding='utf-8')
    df_train['id'] = df_train['id'].apply(str)
    
    df_test = pd.read_csv('test.csv', encoding='utf-8')
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

############### LOADING MODELS ###############
try:
    word = wordVec_model['word']
    print 'using loaded model.....'
except:
    wordVec_model = gensim.models.KeyedVectors.load_word2vec_format("..\..\..\GoogleNews-vectors-negative300.bin.gz",binary=True)


############### EXTRACTING FEATURES ###############

sent_A = q1_train
sent_B = q2_train

model = wordVec_model
def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
#    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

norm_model = model
norm_model.init_sims(replace=True)

def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
#    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)

missing_wrds = 0
total_wrds = 0

def sent2vec(s):
    global total_wrds, missing_wrds
    words = str(s).lower().decode('utf-8')
    words = tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        total_wrds += 1
        try:
            M.append(model[w])
        except:
            missing_wrds += 1
            print 'Word not present in model:', w
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

sent_A_vectors  = np.zeros((len(sent_A), 300))
for i, q in enumerate(sent_A):
    sent_A_vectors[i, :] = sent2vec(q)

sent_B_vectors  = np.zeros((len(sent_B), 300))
for i, q in enumerate(sent_B):
    sent_B_vectors[i, :] = sent2vec(q)

print 'Missing Words :', missing_wrds, '\nOut of:', total_wrds

fuzz_qratio = []
fuzz_Wratio = []
fuzz_partial_ratio = []
fuzz_partial_token_set_ratio = []
fuzz_partial_token_sort_ratio = []
fuzz_token_set_ratio = []
fuzz_token_sort_ratio = []
wmd_list = []
norm_wmd_list = []
cosine_distance = []
cityblock_distance = []
jaccard_distance = []
canberra_distance = []
euclidean_distance = []
minkowski_distance = []
braycurtis_distance = []
skew_q1vec = []
skew_q2vec = []
kur_q1vec = []
kur_q2vec = []

for i in  range(len(sent_A)):
    fuzz_qratio.append(fuzz.QRatio(sent_A[i], sent_B[i]))
    fuzz_Wratio.append(fuzz.WRatio(sent_A[i], sent_B[i]))
    fuzz_partial_ratio.append(fuzz.partial_ratio(sent_A[i], sent_B[i]))
    fuzz_partial_token_set_ratio.append(fuzz.partial_token_set_ratio(sent_A[i], sent_B[i]))
    fuzz_partial_token_sort_ratio.append(fuzz.partial_token_sort_ratio(sent_A[i], sent_B[i]))
    fuzz_token_set_ratio.append(fuzz.token_set_ratio(sent_A[i], sent_B[i]))
    fuzz_token_sort_ratio.append(fuzz.token_sort_ratio(sent_A[i], sent_B[i]))
    wmd_list.append(wmd(sent_A[i], sent_B[i]))
    norm_wmd_list.append(norm_wmd(sent_A[i], sent_B[i]))

cosine_distance = [cosine(x, y) for (x, y) in zip(np.nan_to_num(sent_A_vectors),
                                                          np.nan_to_num(sent_B_vectors))]

cityblock_distance = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(sent_A_vectors),
                                                          np.nan_to_num(sent_B_vectors))]

jaccard_distance = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(sent_A_vectors),
                                                          np.nan_to_num(sent_B_vectors))]

canberra_distance = [canberra(x, y) for (x, y) in zip(np.nan_to_num(sent_A_vectors),
                                                          np.nan_to_num(sent_B_vectors))]

euclidean_distance = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(sent_A_vectors),
                                                          np.nan_to_num(sent_B_vectors))]

minkowski_distance = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(sent_A_vectors),
                                                          np.nan_to_num(sent_B_vectors))]

braycurtis_distance = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(sent_A_vectors),
                                                          np.nan_to_num(sent_B_vectors))]

skew_q1vec = [skew(x) for x in np.nan_to_num(sent_A_vectors)]
skew_q2vec = [skew(x) for x in np.nan_to_num(sent_B_vectors)]
kur_q1vec = [kurtosis(x) for x in np.nan_to_num(sent_A_vectors)]
kur_q2vec = [kurtosis(x) for x in np.nan_to_num(sent_B_vectors)]


############### SAVING FEATURES AND LABELS ###############

data = pd.DataFrame()

data['fuzz_qratio'] = fuzz_qratio
data['fuzz_WRatio'] = fuzz_Wratio
data['fuzz_partial_ratio'] = fuzz_partial_ratio
data['fuzz_partial_token_set_ratio'] = fuzz_partial_token_set_ratio
data['fuzz_partial_token_sort_ratio'] = fuzz_partial_token_sort_ratio
data['fuzz_token_set_ratio'] = fuzz_partial_token_set_ratio
data['fuzz_token_sort_ratio'] = fuzz_partial_token_sort_ratio
data['WMD_list'] = wmd_list
data['norm_wmd'] = norm_wmd_list
data['cosine_distance'] = cosine_distance
data['cityblock_distance'] = cityblock_distance
data['jaccard_distance'] = jaccard_distance
data['canberra_distance'] = canberra_distance
data['euclidean_distance'] = euclidean_distance
data['minkowski_distance'] = minkowski_distance
data['braycurtis_distance'] = braycurtis_distance
data['skew_q1vec'] = skew_q1vec
data['skew_q2vec'] = skew_q2vec
data['kur_q1vec'] = kur_q1vec
data['kur_q2vec'] = kur_q2vec

np.save('../Extracted_Features/numeric', data)
np.save('../Extracted_Features/label', index)

############### SAVING TIME TAKEN AND XGBOOST BEST LOG LOSS ###############

end_time = time()

bst_loss = xgb_bst(data, index)

new_time_loss = [bst_loss, (end_time-start_time)]

try:
    time_loss = np.load('../Extracted_Features/Time_Loss_Numeric.npy')
except:
    time_loss = np.array([])

try:
    time_loss = np.vstack((time_loss, new_time_loss))
except:
    time_loss = list(time_loss)
    time_loss.append(new_time_loss)

np.save('../Extracted_Features/Time_Loss_Numeric', time_loss)