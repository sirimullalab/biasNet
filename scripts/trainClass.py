# ********************************************
# openDMPK Model Training (Classification)****
# ********************************************
# Last modified: 4/24/2020

from hypopt import GridSearch
import joblib
import os, sys, glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import json
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import itertools

class Train_models: 
    def __init__(self, train=None,test=None,validation=None):
        self.train = train
        self.test = test
        self.val = validation
    
        self.random_state = 1
        
        self.reports = {}

    def split_data(self):
        
        if self.test is None and self.val is None:
            print('Using Combined train and test sests')
            x = self.train[:,:-1]
            y = self.train[:,-1]
            
            self.x_train, self.x_test, self.y_train, self.y_test = \
                    train_test_split(x, y, test_size = 0.25, random_state = self.random_state)
            self.x_val = None
            self.y_val = None
            return True

        elif self.test is not None and self.val is None:
            print('Using separate train and test sets')
            
            self.x_train = self.train[:,:-1]
            self.y_train = self.train[:,-1]
            
            self.x_test = self.test[:,:-1]
            self.y_test = self.test[:,-1]
            
            self.x_val = None
            self.y_val = None
            return True
        
        else:
            print('Using separate train, test, and validation sets')

            self.x_train = self.train[:,:-1]
            self.y_train = self.train[:,-1]
            
            self.x_test = self.test[:,:-1]
            self.y_test = self.test[:,-1]

            self.x_val = self.val[:,:-1]
            self.y_val = self.val[:,-1]
            return True
    
    def write_results(self):
        # Save the report
        with open(os.path.join(REPORTS, dir_name, file_name+'.json'), 'w') as f:
            json.dump(self.reports, f)
            print('results saved') 

    def get_results(self, mdl):
        # TODO: add PR_AUC metric
        y_pre = mdl.predict(self.x_test)
        y_pro = mdl.predict_proba(self.x_test)[:, 1]
        
        tn, fp, fn, tp = metrics.confusion_matrix(self.y_test, y_pre).ravel()
        
        SE = float(tp)/(tp+fn)
        SP = float(tn)/(tn+fp)
        PPV = float(tp)/(tp+fp)
        
        results = {}
        results['roc_auc'] = metrics.roc_auc_score(self.y_test, y_pro)
        report_path= os.path.join(REPORTS, file_name+'.rf')
        results['accuracy'] = metrics.accuracy_score(self.y_test, y_pre)
        results['f1_score'] = metrics.f1_score(self.y_test, y_pre)# average='binary' by default
        results['Recall'] = metrics.recall_score(self.y_test, y_pre)
        results['Precison'] = metrics. precision_score(self.y_test, y_pre)
        results['SE'] = SE
        results['SP'] = SP
        results['PPV'] = PPV
        results['cohen_kappa'] = metrics.cohen_kappa_score(self.y_test, y_pre)
        results['mcc'] = metrics.matthews_corrcoef(self.y_test, y_pre)

        if self.val is not None:
            results['data_info'] = {"train_count": len(self.x_train), \
                "test_count": len(self.x_test), "val_count":len(self.val)}
        else:
            results['data_info'] = {"train_count": len(self.x_train), "test_count": len(self.x_test)}

        return results
    
    def rf_classifier(self):
        
        clf = RandomForestClassifier(random_state = self.random_state)
        
        params = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=800, num=8)],\
                        'max_features':['auto', 'sqrt', 'log2']}
        
        mdl = GridSearch(model=clf, param_grid=params)
        print('Fitting Rf')
        
        mdl.fit(self.x_train, self.y_train, self.x_val, self.y_val) #, scoring = 'f1') did not work (may be becuase of sklearn 0.22.1)
         
        print('getting report') 
        self.reports['rf'] = self.get_results(mdl)
        
        model_path = os.path.join(MODELS, dir_name, file_name+'.rf')
        
        joblib.dump(mdl, model_path)
    
    def   xgb_classifier(self):
        
        clf = xgb.XGBClassifier(random_state=self.random_state)
        
        params = {
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
            'n_estimators': [int(x) for x in np.linspace(start=50, stop=800, num=16)],
            #'colsample_bytree': [i/10.0 for i in range(3,11)]
            }

        mdl = GridSearch(model=clf, param_grid=params)
        print('Fitting XGBoost')

        mdl.fit(self.x_train, self.y_train)

        mdl.fit(self.x_train, self.y_train, self.x_val, self.y_val)

        self.reports['xgb'] = self.get_results(mdl)
                
        model_path = os.path.join(MODELS, dir_name, file_name+'.xgb')
        joblib.dump(mdl, model_path)

    def svm_classifier(self):
        
        clf = SVC(random_state = self.random_state, probability = True)

        # Create the parameter grid
        params = {'C': [1, 5, 10], 'kernel':['linear'],\
                'gamma': ['auto', 'scale', 0.1, 1.0, 10, 100, 1000]}
        
       
        mdl = GridSearch(model=clf, param_grid=params)

        mdl.fit(self.x_train, self.y_train, self.x_val, self.y_val)

        self.reports['svm'] = self.get_results(mdl)
        model_path = os.path.join(MODELS, dir_name, file_name+'.svm')
        joblib.dump(mdl, model_path)

    def mlp_classifier(self):
        def get_hidden_layers():
            
            x = [64, 128, 256]
            hl = []

            for i in range(1, len(x)):
                hl.extend([p for p in itertools.product(x, repeat=i+1)])

            return hl
        
        clf = MLPClassifier(solver='adam', alpha=1e-5, early_stopping=True, \
                            random_state=self.random_state)
        
        hidden_layer_sizes = get_hidden_layers()
        params = {'hidden_layer_sizes': hidden_layer_sizes}

        mdl = GridSearch(model=clf, param_grid=params)
        
        mdl.fit(self.x_train, self.y_train, self.x_val, self.y_val)

        self.reports['mlp'] = self.get_results(mdl)
        
        model_path = os.path.join(MODELS, dir_name, file_name+'.mlp')
        joblib.dump(mdl, model_path)

    def train_all(self):
        
        #self.rf_classifier()
        
        #self.xgb_classifier()
        
        #self.svm_classifier()
       
        self.mlp_classifier()
        
        self.write_results()

if __name__=="__main__":

    MODELS = 'models_cano'
    REPORTS = 'reports_cano'

    if not os.path.isdir(MODELS):
        os.mkdir(MODELS)
    if not os.path.isdir(REPORTS):
        os.mkdir(REPORTS)    
    
    file_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
    
    dir_name = 'biasnet' 
    arguments = len(sys.argv) - 1

    if arguments ==3:
        train = np.load(sys.argv[1])
        test = np.load(sys.argv[2])
        valid = np.load(sys.argv[3])
        t = Train_models(train, test, valid)
    elif arguments == 2:
        train = np.load(sys.argv[1])
        test = np.load(sys.argv[2])
        valid = None
        t = Train_models(train, test, valid)
    else:
        train = np.load(sys.argv[1])
        test = None
        valid = None
        t = Train_models(train, test, valid)

    if not t.split_data():
        print('Features files are not present')
        exit()
    t.train_all()
