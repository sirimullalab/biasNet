import joblib
import numpy as np
import pandas as pd
import glob, os, sys

models = glob.glob('../models/*')

for m in models:
    model_name = os.path.basename(m).split('.')[-1]
    if model_name == 'mlp':
        data = np.load('../numpyFiles/b_g_labels_clean_test_lecfp4_fp.npy')
    elif model_name =='xgb' or model_name == 'rf':
        data = np.load('../numpyFiles/b_g_labels_clean_test_avalon_fp.npy')
    elif model_name == 'svm':
        data = np.load('../numpyFiles/b_g_labels_clean_test_ecfp2_fp.npy')
    else:
        print('File not found')
        pass
    X_true = data[:, :-1]
    print(X_true.shape)
    y_true = data[:, -1]
    load_model = joblib.load(m)
    y_pred = load_model.predict(X_true)
    y_prob = load_model.predict_proba(X_true)
    y_pred_label = y_pred.tolist()
    load_test = pd.read_csv('../data/b_g_labels_clean_test_stand.csv')
    load_test.insert(len(load_test.columns), 'pred_label_'+model_name.lower(), y_pred_label)
    print('Done')
    load_test.insert(len(load_test.columns), 'pred_label_'+model_name.lower()+'_prob', list(y_prob))
    load_test.to_csv('../results/prediction_results2/'+\
                     'b_g_labels_clean_test_stand_'+\
                     model_name.lower()+'_pred.csv', index=False)
