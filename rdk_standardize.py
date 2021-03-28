import os,sys,re,argparse,logging
import pandas as pd
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.MolStandardize

def StdSmiles(smi):
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

#############################################################################
if __name__ == "__main__":
    df = pd.read_csv('data/b_g_labels_clean_test.csv')
    

    smis = df['SMILES'].to_list()
    smis_stds = []
    not_found = [] 
    for smi in smis:
        try:
            std = StdSmiles(smi)
        except:
            std = '-'
            not_found.append(smi)
            pass
        smis_stds.append(std)
    print('Total number of smiles that could not be converted', len(not_found))
    df.insert(1, 'SMILES_std', smis_stds)
    df.to_csv('data/b_g_labels_clean_test_stand.csv', index=False)