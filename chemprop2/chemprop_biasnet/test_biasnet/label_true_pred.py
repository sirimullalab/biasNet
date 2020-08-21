# coding: utf-8
import glob
import json, os, sys, pandas as pd
files = glob.glob('results/*.csv')
with open('trueLabel.json', 'r') as f:
    labels = json.load(f)
for f in files:
    filename,_ = os.path.splitext(os.path.basename(f))
    filename2 = 'data_clean'#filename.split('_')[0]
    print(filename2)
    df = pd.read_csv(f)
    tLabel= []
    pLabel= []
    print('working on', f)
    for i in range(len(df)):
        tLabel.append(labels[filename2][df['Canonical_Smiles'][i]])
        if df['Label'][i] >= 0.5:
            pLabel.append(1)
        else:
            pLabel.append(0)
    df['pLabel']=pLabel
    df['tLabel']=tLabel
    df.to_csv('results_with_trueLabels/'+filename+'.csv', index=False)
   
