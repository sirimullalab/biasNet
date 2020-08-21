import sys,os,glob
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
#from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors


class FeaturesGeneration:
    
    def __init__(self):
        self.fpdict = {}
        self.fp_name = 'lecfp4'
        self.nbits = 1024
        self.longbits = 16384
        self.fingerprints = []

    def get_features(self, smiles):

        self.fpdict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=self.longbits)
 #       self.fpdict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, self.nbits)
        
        m = Chem.MolFromSmiles(smiles)
    
        if m is None:
            raise ValueError(smiles)
            return None 
        fp = self.fpdict[self.fp_name](m)
        bit_array = np.asarray(fp)

        self.fingerprints.append(bit_array)
        fp_array = ( np.asarray((self.fingerprints), dtype=object) )
        X = np.vstack(fp_array).astype(np.float32)
        
        return X
