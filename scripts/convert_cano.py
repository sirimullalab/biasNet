import rdkit, sys, os, glob
from rdkit import Chem
import pandas as pd

def smiles_to_cano(csv_folder):
    
    for csv in glob.glob(csv_folder+'/*.csv'):
        file_name = os.path.basename(csv)    
        df = pd.read_csv(csv)
        cano = []
        problems = []
        for i in range(len(df)):
            try:
                mol = Chem.MolFromSmiles(df['SMILES'][i])
                can_smi = Chem.MolToSmiles(mol, True)
            except:
                problems.append(i)
                return None

            cano.append(can_smi)
        newdf=df 
        newdf.insert(1, 'Canonical_Smiles', cano)
        newdf.reset_index(drop = True, inplace = True)
        newdf.drop(problems, inplace = True)
        newdf.reset_index(drop = True, inplace = True)
        newdf.to_csv('data_clean_cano/'+file_name, index=False)
        
if __name__=='__main__':
    
    smiles_to_cano(sys.argv[1])
