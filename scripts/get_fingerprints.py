# This program can be used to generate total 10 different Fingerprints or ..
# Molecular Descriptors from RDKit as well as MayaChemTools
# For rdkit documentation: Specially for rdk_fp (RDKitfingerprint, size 2048): ..
# See: http://rdkit.org/docs/source/rdkit.Chem.rdmolops.html
# See: https://www.rdkit.org/docs/source/rdkit.ML.Descriptors.MoleculeDescriptors.html (For molecule descriptors)
# See: https://sourceforge.net/p/rdkit/mailman/message/30087006/ (For molecule descriptors)
# Note: rdk_fp and molecule_descriptors are not same.

# Usage:   python new_fp_des.py folder_containing_csv_files
# Note: morgan and ecfp4 are same (i think so, ecfp4 is removed)
# Last modified: 4/24/2020

import math
from rdkit.Chem import AllChem, Descriptors
from rdkit import Chem
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
import pandas as pd
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import RDKFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
#from rdkit.Chem.Fingerprints import FingerprintMols
import os, glob
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import MinMaxScaler
#from rdkit.Chem.AtomPairs import Pairs
import tempfile
import shutil
from pathlib import Path

class Features_Generations:
    def __init__(self, csv_path, features_type):
        self.csv_path = csv_path
        self.features_type = features_type
        self.temp_dir = tempfile.mkdtemp()
        
    def toSDF(self, smiles):
        # Get mol format of smiles
        mol = Chem.MolFromSmiles(smiles)
        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)
        mol.SetProp("smiles", smiles)
        
        w = Chem.SDWriter(os.path.join(self.temp_dir, "temp.sdf"))
        w.write(mol)
        w.flush()

    def _cleanup(self):
        shutil.rmtree(self.temp_dir)

    def toTPATF(self):
        features = []
        script_path = "../../Downloads/mayachemtools/bin/TopologicalPharmacophoreAtomTripletsFingerprints.pl"
          
        # Now generate the TPATF features
        # Check if the sdf file exists
        
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")): return None
        command = "perl " + script_path + " -r " + os.path.join(self.temp_dir, "temp") + " --AtomTripletsSetSizeToUse FixedSize -v ValuesString -o " + os.path.join(self.temp_dir, "temp.sdf")
        os.system(command)
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"','')
                    features = [int(i) for i in line.split(" ")]
        return features
    
    def toTPAPF(self):
        script_path = "../../mayachemtools/bin/TopologicalPharmacophoreAtomPairsFingerprints.pl"
        
        # Generate TPAPF features
        # Check if the sdf file exists
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")): return None
        command = "perl " + script_path + " -r " + os.path.join(self.temp_dir, "temp") + " --AtomPairsSetSizeToUse FixedSize -v ValuesString -o " + os.path.join(self.temp_dir, "temp.sdf")
        os.system(command)
        
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():
                if "Cmpd" in line:
                    line = line.split(';')[5].replace('"','')
                    features = [int(i) for i in line.split(" ")]
        
        return features

    def toPHYC(self):
        script_path = "../../mayachemtools/bin/CalculatePhysicochemicalProperties.pl"
        
        # Now generate the PHYS features
        # Check if the sdf file exists
        if not os.path.isfile(os.path.join(self.temp_dir, "temp.sdf")): return None
        command = "perl " + script_path + " -r " + os.path.join(self.temp_dir, "temp")+" -o " + os.path.join(self.temp_dir,"temp.sdf")
        os.system(command)
        
        with open(os.path.join(self.temp_dir, "temp.csv"), 'r') as f:
            for line in f.readlines():

                if "Cmp" in line:
                    line = line.replace('"','')
                    line = ','.join(line.split(',')[1:])
                    features = [float(i) for i in line.split(",")]

        return features
   

    def tpatf_fp(self):
        df= pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()
        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                self.toSDF(smiles_list[i])  
                # features = fg.toTPATF()
                features = self.toTPATF()
                
                fingerprints.append(features)
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass

        # Clean up the temporary files
        self._cleanup()
        
        for i in range(len(df)):
            if df['Label'].isnull().iloc[i]:
                not_found.append(i)
        print('not_found', len(not_found))
        df.drop(not_found, axis=0,inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        
        # Drop rows from array where FP not generated
        X = np.delete(fp_array, not_found, axis=0)
        
        X = np.vstack(X).astype(np.float32)                 
        
        print('Input shape: {}'.format(X.shape))
        
        #Concatenating input and output array
        final_array = np.concatenate((X, Y), axis=1)
        
        return final_array  
                           
    def tpapf_fp(self):
        df= pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()

        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                print('to_sdf is called')
                self.toSDF(smiles_list[i])  
                
                print('came back from tosdf and getfeatuers_calling')
                #features = fg.toTPAPF()
                features = self.toTPAPF()
                
                fingerprints.append(features)
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
            
        # Clean up the temporary files            
        self._cleanup()
        
        for i in range(len(df)):
            if df['Label'].isnull().iloc[i]:
                not_found.append(i)
        print('not_found', len(not_found))
        df.drop(not_found, axis=0,inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        ##
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
    
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        
        #Drop rows from array where FP not generated
        X = np.delete(fp_array, not_found, axis=0)
        #Save as dytpe = np.float32
        X = np.vstack(X).astype(np.float32) 
        
        print('Input shape: {}'.format(X.shape))
        #Concatenating input and output array
        final_array = np.concatenate((X, Y), axis=1)
        return final_array
    
    def phyc(self):
        df= pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()

        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                self.toSDF(smiles_list[i])  
                #features = fg.toPHYC()
                features = self.toPHYC()
                fingerprints.append(features)
            
            except:
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
            
        # Clean up the temporary files    
        self._cleanup()
        
        for i in range(len(df)):
            if df['Label'].isnull().iloc[i]:
            #a=df['Label'][i]
            #if np.isnan(a):
            #if pd.isnull(np.array([int(a)], dtype=float)):
                not_found.append(i)
        print('not_found', len(not_found))
        df.drop(not_found, axis=0,inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        #Drop rows from array where FP not generated
        X = np.delete(fp_array, not_found, axis=0)   
        X = np.vstack(X).astype(np.float32)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)
        return final_array
    
    def morgan_fp(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()
        
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
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
        ##
        #labelencoder = LabelEncoder()                       
        #Y = labelencoder.fit_transform(df['Label'].values)
        #Y = Y.reshape(Y.shape[0],1)
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)          
        X = np.vstack(X).astype(np.float32)                 
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)        
        
        return final_array
    
    
    def ecfp2_fp(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()
        
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=1024)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
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
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)
        X = np.vstack(X).astype(np.float32)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)
        return final_array
    
    def ecfp6_fp(self):
        
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()
        
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        for i in range(len(df)):
            if df['Label'].isnull().iloc[i]:
            #a=df['Label'][i]
            #if np.isnan(a):
            #if pd.isnull(np.array([int(a)], dtype=float)):
                not_found.append(i)
        print('not_found', len(not_found))
        df.drop(not_found, axis=0,inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)  
        X = np.vstack(X).astype(np.float32)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)
        
        return final_array
    
    
    def maccs_fp(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()
        
        fingerprints = []
        not_found = []

        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = MACCSkeys.GenMACCSKeys(mol)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
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
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0) 
        X = np.vstack(X).astype(np.float32)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)
        return final_array
    
    
    def avalon_fp(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()
        
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=512)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
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
        
                
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)   
        X = np.vstack(X).astype(np.float32)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)
        
        return final_array
    
    
    def rdk_fp(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()
    
        fingerprints = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                mol = Chem.MolFromSmiles(smiles_list[i])
                fp = RDKFingerprint(mol, nBitsPerHash=1)
                bits_array = (np.fromstring(fp.ToBitString(),'u1') - ord('0'))
                fingerprints.append(bits_array)
            
            except:
                
                fingerprints.append(np.nan)
                not_found.append(i)
                pass
    
        for i in range(len(df)):
            if df['Label'].isnull().iloc[i]:
            #a=df['Label'][i]
            #if np.isnan(a):
            #if pd.isnull(np.array([int(a)], dtype=float)):
                not_found.append(i)
        print('not_found', len(not_found))
        df.drop(not_found, axis=0,inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        ##
        #labelencoder = LabelEncoder()
        #Y = labelencoder.fit_transform(df['Label'].values)
        #Y = Y.reshape(Y.shape[0],1)
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
        
        fp_array = ( np.asarray((fingerprints), dtype=object) )
        X = np.delete(fp_array, not_found, axis=0)
        X = np.vstack(X).astype(np.float32)
        
        print('Input shape: {}'.format(X.shape))
        
        final_array = np.concatenate((X, Y), axis=1)
        return final_array
    
    
    def molecule_descriptors(self):
        df = pd.read_csv(self.csv_path)
        smiles_list = df['Canonical_Smiles'].tolist()
        
        descriptors = []
        not_found = []
        
        for i in tqdm(range(len(smiles_list))):
            try:
                
                calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
                mol = Chem.MolFromSmiles(smiles_list[i])
                ds = calc.CalcDescriptors(mol)
                ds = list(ds)
                max_value = max(ds)
                if max_value > 10**20:
                    not_found.append(i)
                for k in ds:
                    if math.isnan(k):
                        print('yes nan')
                        not_found.append(i)
                        break
                ds = np.asarray(ds)
                descriptors.append(ds)
            
            except:
                
                descriptors.append(np.nan)
                not_found.append(i)
                print('Nan is added')
        for i in range(len(df)):
            if df['Label'].isnull().iloc[i]:
                not_found.append(i)
        print('not_found', len(not_found))
        df.drop(not_found, axis=0,inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        Y = np.array([f for f in df.Label.values])
        Y = Y.reshape(Y.shape[0],1)

        print('Output shape: {}'.format(Y.shape))
    
        fp_array = ( np.asarray((descriptors), dtype=object) )
        #Drop rows from array where Descriptor not generated
        X = np.delete(fp_array, not_found, axis=0)
        X = np.vstack(X).astype(np.float32)                 
    
        print('Input shape: {}'.format(X.shape))
    
        final_array = np.concatenate((X, Y), axis=1)
        return final_array


def main():
    if fp_name == 'morgan_fp':
        numpy_file = fg.morgan_fp()
    elif fp_name == 'maccs_fp':
        numpy_file = fg.maccs_fp()
    elif fp_name == 'avalon_fp':
        numpy_file = fg.avalon_fp()
    elif fp_name == 'rdk_fp':
        numpy_file = fg.rdk_fp()
    elif fp_name == 'molecule_descriptors':
        numpy_file = fg.molecule_descriptors()
    elif fp_name == 'ecfp2_fp':
        numpy_file = fg.ecfp2_fp()
    elif fp_name == 'ecfp6_fp':
        numpy_file = fg.ecfp6_fp()
    elif fp_name == 'tpatf_fp':
        numpy_file = fg.tpatf_fp()
    elif fp_name == 'tpapf_fp':
        numpy_file = fg.tpapf_fp()
    elif fp_name == 'phyc_descriptors':
        numpy_file = fg.phyc()
    else:
        print('FingerPrint is not available, Please check the FingerPrint name!')
        exit()
    # Saving numpy file of fingerprints or molecule_descriptors
    
    np.save(dest+'/'+file_name+'_'+fp_name+'.npy', numpy_file)
    print('Features saved')


if __name__ == "__main__":

    dest = 'features_cano'

    if not os.path.isdir(dest): os.mkdir(dest)

    workdir = sys.argv[1]
    dir_name = os.path.dirname(workdir)
    dir_name = os.path.basename(dir_name)

    files = glob.glob(workdir+'/*.csv')
    fp_list = ['avalon_fp']
    #fp_list = ['morgan_fp', 'avalon_fp', 'rdk_fp', 'molecule_descriptors', \
   #         'tpatf_fp', 'phyc_descriptors', 'tpapf_fp', 'ecfp6_fp', 'ecfp2_fp', 'maccs_fp']
    
    for fp_name in fp_list:
        for f in files:
            file_name = os.path.basename(f)[:-4]
            fg = Features_Generations(f, fp_name)
            main()
