# ********************************************
# openDMPK Model Training (Classification)****
# ********************************************
# Last modified: 4/24/2020

#from hypopt import GridSearch
import joblib
import os, sys, glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import json
#import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import itertools
import pandas as pd


class CalculateResults(object):
    def __init__(self, csv):
        self.csv = csv
        self.final_result = {}
        filename2,_ = os.path.splitext(os.path.basename(self.csv))
        self.filename = filename2.split('_')[0]
    def get_results(self):
        # TODO: add PR_AUC metric

        df = pd.read_csv(self.csv)
        y_pro = df['Label'].to_list()
        
        y_pre = df['pLabel'].to_list()
        y_test = df['tLabel'].to_list()

        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pre).ravel()
        
        SE = float(tp)/(tp+fn)
        SP = float(tn)/(tn+fp)
        PPV = float(tp)/(tp+fp)
        
        results = {}
        results['roc_auc'] = metrics.roc_auc_score(y_test, y_pro)
        results['accuracy'] = metrics.accuracy_score(y_test, y_pre)
        results['f1_score'] = metrics.f1_score(y_test, y_pre)# average='binary' by default
        results['Recall'] = metrics.recall_score(y_test, y_pre)
        results['Precison'] = metrics. precision_score(y_test, y_pre)
        results['SE'] = SE
        results['SP'] = SP
        results['PPV'] = PPV
        results['cohen_kappa'] = metrics.cohen_kappa_score(y_test, y_pre)
        results['mcc'] = metrics.matthews_corrcoef(y_test, y_pre)

        self.final_result[self.filename]=results
        out = json.dumps(self.final_result)

        open('dmpnn_results_json/'+self.filename+'.json', 'w').write(out)

if __name__ == '__main__':
    files = glob.glob('results_with_trueLabels/*.csv')
    for csvFile in files:
        #print(csvFile)
        cr = CalculateResults(csvFile)
        cr.get_results()
