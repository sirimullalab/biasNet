# ********************************* #
# Govinda KC                        #
# UTEP, Computational Science       #
# Last modified: 8/19/2020          #
# ********************************* #

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
import os,sys,re,argparse,logging
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.MolStandardize

class FeaturesGeneration:
    
    def __init__(self):
        self.fpdict = {}
        self.fp_name = 'lecfp4'
        self.nbits = 1024
        self.longbits = 16384
        self.fingerprints = []

    def StdSmiles(self, smi):
        def find_norms():
            norms = list(rdkit.Chem.MolStandardize.normalize.NORMALIZATIONS)
            for i in range(len(norms)-1, 0, -1):
                norm = norms[i]
                if norm.name == "Sulfoxide to -S+(O-)-":
                    del(norms[i])
            norms.append(rdkit.Chem.MolStandardize.normalize.Normalization("[S+]-[O-] to S=O", \
              "[S+:1]([O-:2])>>[S+0:1](=[O-0:2])"))
            
            s = rdkit.Chem.MolStandardize.Standardizer(
                normalizations = norms,
                max_restarts = rdkit.Chem.MolStandardize.normalize.MAX_RESTARTS,
                prefer_organic = rdkit.Chem.MolStandardize.fragment.PREFER_ORGANIC,
                acid_base_pairs = rdkit.Chem.MolStandardize.charge.ACID_BASE_PAIRS,
                charge_corrections = rdkit.Chem.MolStandardize.charge.CHARGE_CORRECTIONS,
                tautomer_transforms = rdkit.Chem.MolStandardize.tautomer.TAUTOMER_TRANSFORMS,
                tautomer_scores = rdkit.Chem.MolStandardize.tautomer.TAUTOMER_SCORES,
                max_tautomers = rdkit.Chem.MolStandardize.tautomer.MAX_TAUTOMERS
                )
            return s
        
        s = find_norms()

        mol1 = rdkit.Chem.MolFromSmiles(smi)
        mol2 = s.standardize(mol1) if mol1 else None
        smi_std = rdkit.Chem.MolToSmiles(mol2, isomericSmiles=False) if mol2 else None

        return smi_std

    def get_fingerprints(self, smiles): # features--> fingerprints
        smiles = self.StdSmiles(smiles)

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
