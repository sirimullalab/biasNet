# coding: utf-8
import os
import pandas as pd
import glob
import json
trueDictn = {}
wd = '../../../../projects_github/biasNet'
dirs = os.listdir(wd)
checkDup = {}

for d in dirs:
    if d == 'data_clean':
            
        smiles_label = {}
        df = pd.read_csv(os.path.join(wd,d,'b_g_labels_clean_test.csv'))
        seen = []
        repeat=[]
        smiles=df['Canonical_Smiles'].to_list()
        Label = df['Label'].to_list()
        for i,j in zip(smiles, Label):
            if i in seen:
                repeat.append(i)
                continue
            else:
                smiles_label[i]=j
                seen.append(i)
        if len(df)==len(smiles_label):
            checkDup[d]='True'
        else:
            checkDup[d]=repeat
        
        trueDictn[d]=smiles_label
    out = json.dumps(trueDictn)
    open('trueLabel.json', 'w').write(out)
    out2 = json.dumps(checkDup)
    open('check_duplicates.json', 'w').write(out2)
            
        
