# ********************************* #
# Govinda KC                        #
# UTEP, Computational Science       #
# Last modified: 8/19/2020          #
# ********************************* #

# Usage: python app.py --smiles SMILES ( eg. python app.py --smiles "CC(O)CO" )

import os, joblib, json
import argparse
from tqdm import tqdm
from features import FeaturesGeneration
from pprint import pprint
from rdkit import Chem

class biasNet:

    MODELS_DIR = os.path.join('models')

    def __init__(self):

        self.check_smiles()
        self.load_models()
    
    def check_smiles(self):
        mol = Chem.MolFromSmiles(input_smiles)
        
        if len(input_smiles) == 0:
            print(f'Given SMILES: {input_smiles} can not be Predicted')
            exit(1)
        elif not mol:
            print(f'Given SMILES: {input_smiles} can not be Predicted')
            exit(1)

    def load_models(self):
        
        with open('models.txt', 'r') as f:
            models = f.read().splitlines()

        self.model_names = [model_path.split('_')[0] for model_path in models]
        self.models = [joblib.load(os.path.join(self.MODELS_DIR, model_path)) for model_path in models]


    def predict(self, smiles):
        
        fg = FeaturesGeneration()

        features = fg.get_features(smiles)
        
        final_results = {}
        model_result = {}
        
        # First index ->  probability that the data belong to class 0,
        # Second index ->  probability that the data belong to class 1.
        
        for model_name, model in tqdm(zip(self.model_names, self.models)):
            label_zero = model.predict_proba(features)[0][0].round(3)
            label_one = model.predict_proba(features)[0][1].round(3)

            if label_one >= 0.5:
                model_result['Prediction'] = 'B-Arrestin'
                model_result['Confidence'] = label_one
            else:
                model_result['Prediction'] = 'G-Protein'
                model_result['Confidence'] = label_zero

        final_results[smiles] = model_result
        
        pprint(final_results)
         
        with open('biasnet_results.json', 'w') as json_file:
            json.dump(final_results, json_file)
        
        print('Result file is saved')

if __name__=='__main__':
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description = "BiasNet")
    ap.add_argument("-s", "--smiles", action = 'store', dest = 'smiles', 
            type = str, required = True, help = "SMILES string")
    
    args = vars(ap.parse_args())
    input_smiles = args['smiles']
    input_smiles = input_smiles.strip()    
    
    biasnet = biasNet()
    
    biasnet.predict(input_smiles)

