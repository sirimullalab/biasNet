# module to calculate a fingerprint (Specially longbits fp) from SMILES
# Last modified: 4/24/2020
# Usage: python long_fp.py folder_contain_csv_files

#from sklearn.preprocessing import LabelEncoder
import sys,os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
#from rdkit.Chem.ChemicalFeatures import BuildFeatureFactory
from rdkit.Chem import rdMolDescriptors
import glob
from pathlib import Path
nbits = 1024
longbits = 16384

# dictionary
fpdict = {}

fpdict['ecfp0_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpdict['ecfp2_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpdict['ecfp4_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpdict['ecfp6_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpdict['ecfc0_fp'] = lambda m: AllChem.GetMorganFingerprint(m, 0)
fpdict['ecfc2_fp'] = lambda m: AllChem.GetMorganFingerprint(m, 1)
fpdict['ecfc4_fp'] = lambda m: AllChem.GetMorganFingerprint(m, 2)
fpdict['ecfc6_fp'] = lambda m: AllChem.GetMorganFingerprint(m, 3)
fpdict['fcfp2_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpdict['fcfp4_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpdict['fcfp6_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpdict['fcfc2_fp'] = lambda m: AllChem.GetMorganFingerprint(m, 1, useFeatures=True)
fpdict['fcfc4_fp'] = lambda m: AllChem.GetMorganFingerprint(m, 2, useFeatures=True)
fpdict['fcfc6_fp'] = lambda m: AllChem.GetMorganFingerprint(m, 3, useFeatures=True)
fpdict['lecfp4_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpdict['lecfp6_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpdict['lfcfp4_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpdict['lfcfp6_fp'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
fpdict['maccs_fp'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpdict['atom_pair_fp'] = lambda m: Pairs.GetAtomPairFingerprint(m)
fpdict['torsion_fp'] = lambda m: Torsions.GetTopologicalTorsionFingerprintAsIntVect(m)
fpdict['hashap_fp'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpdict['hashtt_fp'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
fpdict['avalon_fp'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
fpdict['lavalon_fp'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpdict['rdk5_fp'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk6_fp'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk7_fp'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)


def CalculateFP(fp_name, smiles):
    m = Chem.MolFromSmiles(smiles)
    
    if m is None:
        raise ValueError(smiles)
        return ""
    return fpdict[fp_name](m)

def get_features(fp_name, csv_path):
    
    file_name = os.path.basename(csv_path)[:-4]
    
    if not os.path.isdir(dest): os.mkdir(dest)
    
    file_path = dest+'/'+file_name+'_'+fp_name+'.npy'
    
    if Path(file_path).is_file():
        
        print(f'{file_path} already exists, moving for next one!')
        return 
    
    fingerprints = []
    not_found = []
    
    df = pd.read_csv(csv_path)      
    smiles_list = df['Canonical_Smiles'].tolist()
    for i in range(len(smiles_list)):
            try:    
                    fp = CalculateFP(fp_name, smiles_list[i])
                    bit_array = np.asarray(fp)
                    fingerprints.append(bit_array)
            except:
                    fingerprints.append(np.nan)
                    not_found.append(i)
                    pass
    
    for i in range(len(df)):
        if df['Label'].isnull().iloc[i]:
            not_found.append(i)
    
    print('not_found', len(not_found))
    df.drop(not_found, axis=0,inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    #**************************************************** 
    # Label encoder is  used for categorical problems   #
    # labelencoder = LabelEncoder()                     #
    # Y = labelencoder.fit_transform(df['Label'].values)#
    #****************************************************
    
    Y = np.array([f for f in df.Label.values])
    Y = Y.reshape(Y.shape[0],1)

    print('Output shape: {}'.format(Y.shape))

    fp_array = ( np.asarray((fingerprints), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)
    X = np.vstack(X).astype(np.float32)

    print('Input shape: {}'.format(X.shape))

    final_array = np.concatenate((X, Y), axis=1)

    np.save(dest+'/'+file_name+'_'+ fp_name +'.npy', final_array)


if __name__ == "__main__":

    dest = 'biasnet_features_new'

    if not os.path.isdir(dest): os.mkdir(dest)

    workdir = sys.argv[1]
    
    files = glob.glob(workdir+'/*.csv')
    
    fp_list = ['rdk5_fp', 'hashap_fp', 'hashtt_fp', 'lavalon_fp', 'lecfp4_fp', 'lfcfp4_fp', 'fcfp2_fp', 'fcfp4_fp', 'fcfp6_fp']
    for fp_name in fp_list:
        for f in files:
            get_features(fp_name, f)
