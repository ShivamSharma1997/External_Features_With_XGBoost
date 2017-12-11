import numpy as np

from utility import xgb_bst

feat_lex = np.load('../Extracted_Features/lex.npy')
feat_read = np.load('../Extracted_Features/read.npy')
feat_numeric = np.load('../Extracted_Features/numeric.npy')

feat = np.hstack((feat_lex,feat_read, feat_numeric))

print 'Best for combined:', xgb_bst(feat)