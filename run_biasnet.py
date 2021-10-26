# *******************************************************#
# Author: Govinda KC - UTEP, Computational Science       #
# Code developed in Sirimulla Research Group             #
# (http://sirimullaresearchgroup.com/)                   #
# Last modified:10/25/2021                               #
# *******************************************************#

# Usage: python run_biasnet.py --smiles SMILES ( eg. python run_biasnet.py --smiles "CC(O)CO" )

import os, joblib, json,sys,re,time,argparse,logging
import argparse 
from tqdm import tqdm 
from features import FeaturesGeneration
from pprint import pprint 
from rdkit import Chem
import rdkit
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from collections import OrderedDict
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MolStandardize
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import pickle
from glob import glob
import numpy as np
import multiprocessing as mp
from time import time
from time import sleep
from requests import get
from random import randint
from datetime import datetime
from urllib import parse
import subprocess
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from pubchempy import Compound, get_compounds, get_synonyms
from func_timeout import func_timeout, FunctionTimedOut
import pandas as pd
import rdkit, shutil
from rdkit.Chem import SmilesMolSupplier, SDMolSupplier, SDWriter, SmilesWriter, MolStandardize, MolToSmiles, MolFromSmiles
import tempfile


pubchem_time_limit = 30  # in seconds
ochem_api_time_limit = 20 # in seconds


def Standardize(stdzr, remove_isomerism, molReader, molWriter):
  n_mol=0; 
  for mol in molReader:
    n_mol+=1
    molname = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
    logging.debug('%d. %s:'%(n_mol, molname))
    mol2 = StdMol(stdzr, mol, remove_isomerism)
    output = rdkit.Chem.MolToSmiles(mol2, isomericSmiles=True) if mol2 else None
    return output
#############################################################################
def MyNorms():
  norms = list(MolStandardize.normalize.NORMALIZATIONS)
  for i in range(len(norms)-1, 0, -1):
   norm = norms[i]
   if norm.name == "Sulfoxide to -S+(O-)-":
     del(norms[i])
  norms.append(MolStandardize.normalize.Normalization("[S+]-[O-] to S=O",
    "[S+:1]([O-:2])>>[S+0:1](=[O-0:2])"))
  logging.info("Normalizations: {}".format(len(norms)))
  return(norms)

#############################################################################
def MyStandardizer(norms):
  stdzr = MolStandardize.Standardizer(
    normalizations = norms,
    max_restarts = MolStandardize.normalize.MAX_RESTARTS,
    prefer_organic = MolStandardize.fragment.PREFER_ORGANIC,
    acid_base_pairs = MolStandardize.charge.ACID_BASE_PAIRS,
    charge_corrections = MolStandardize.charge.CHARGE_CORRECTIONS,
    tautomer_transforms = MolStandardize.tautomer.TAUTOMER_TRANSFORMS,
    tautomer_scores = MolStandardize.tautomer.TAUTOMER_SCORES,
    max_tautomers = MolStandardize.tautomer.MAX_TAUTOMERS
    )
  return(stdzr)

#############################################################################
def StdMol(stdzr, mol, remove_isomerism=False):
  smi = MolToSmiles(mol, isomericSmiles=(not remove_isomerism)) if mol else None
  mol_std = stdzr.standardize(mol) if mol else None
  smi_std = MolToSmiles(mol_std, isomericSmiles=(not remove_isomerism)) if mol_std else None
  logging.debug(f"{smi:>28s} >> {smi_std}")
  return(mol_std)

#############################################################################
def preprocess_smi(smi):
    norms = MolStandardize.normalize.NORMALIZATIONS

    test_smiles = [smi]
    test_label = [1] # dummy list
    temp_dir = tempfile.mkdtemp()
    df = pd.DataFrame(zip(test_smiles, test_label), columns=['SMILES', 'Label'])

    df.to_csv(temp_dir+'/temp_file.csv', index=False)

    try:
        molReader = SmilesMolSupplier(temp_dir+'/temp_file.csv', delimiter=',', smilesColumn=0, nameColumn=1, titleLine=True, sanitize=True)

        molWriter = SmilesWriter(temp_dir+'/temp_outfile.csv', delimiter=',', nameHeader='Name',
        includeHeader=True, isomericSmiles = (True), kekuleSmiles=False)
        stdzr = MyStandardizer(norms)
        stand_smiles = Standardize(stdzr, True, molReader, molWriter)
        shutil.rmtree(temp_dir)
        
        return stand_smiles
    except:
        return None


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

        features = fg.get_fingerprints(smiles)
        
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
                model_result['GPCR_Prediction'] = 'Non-GPCR'
            else:
                model_result['Prediction'] = 'G-Protein'
                model_result['Confidence'] = label_zero
                model_result['GPCR_Prediction'] = 'GPCR'

        final_results[smiles] = model_result
        print('Final results: ', final_results) 
         
        with open('biasnet_results.json', 'w') as json_file:
            json.dump(final_results, json_file, indent=4)
        
        print('Result file is saved')



#########################----GET OCHEM API RESULTS-------###################################
class OchemAPIResults:
    # Ochem URL
    '''
    http://rest.ochem.eu/
    http://rest.ochem.eu/predict?MODELID=536&SMILES=Cc1ccccc1
    '''

    def get_ochem_model_results(self, smiles, model_id):
        try:
            d = func_timeout(ochem_api_time_limit, self.fetch_ochem, args=(smiles, model_id))
            if d[smiles]['response_code'] == 200:
                if model_id == 535:  # logp
                    _val = str(d[smiles]['results']['logPow']['value'])
                    return _val
                elif model_id == 536:  # logs
                    _val = str(d[smiles]['results']['Aqueous Solubility']['value'])
                    return _val

            else:
                return '-'
        except:
            return '-'

    def save_file(self, smi_dict, model_id, save_dir, i, res_code):
        save_path = save_dir + '/smi_' + str(i + 1) + '-response_code_' + str(res_code) \
                    + '-model_id_' + str(model_id) + '.json'

        with open(save_path, 'w') as f:
            json.dump(smi_dict, f, indent=4)

    def fetch_ochem(self, smiles, model_id, save_dir=None):

        # datetime object containing current date and time
        now = datetime.now()
        error_codes = [401, 400, 404]
        requests = 0
        start_time = time()
        total_runtime = datetime.now()
        smi_dict = {}

        s_time = time()
        # SMILES needs to be of HTML format! That's why below line exists-->
        url_smi = parse.quote(smiles)
        smi_dict[smiles] = {'results': -1, 'response_code': -1, 'time_taken': -1,
                            'model_id': -1, 'short_error': -1, 'long_error': -1}

        #         if i % 10 == 0:
        #             print('sleeping for 1 min....')
        #             sleep(randint(60, 80))

        try:

            #######<GET RESPONSE/>#######
            response = get("http://rest.ochem.eu/predict?MODELID={0}&SMILES={1}".format(model_id, url_smi))

            # Monitor the frequency of requests
            requests += 1

            # Pauses the loop between 2 - 4 seconds and marks the elapsed time
            sleep(randint(2, 4))
            current_time = time()
            elapsed_time = current_time - start_time
            print("===================<OchemAPI_RESPONSE>========================")
            print("Total Request:{}; Frequency: {} request/s; Total Run Time: {}".format(requests,
                                                                                         requests / elapsed_time,
                                                                                         datetime.now() - total_runtime))
            #             clear_output(wait=True)

            print("Response Code: ", response.status_code)

            # Throw a warning for non-200 status codes
            if response.status_code in error_codes:
                smi_dict[smiles].update({'results': json.loads(response.text),
                                         'response_code': int(response.status_code),
                                         'time_taken': round((time() - s_time), 3),
                                         'model_id': model_id, 'short_error': 'ERROR',
                                         'long_error': str(response.text)})

                return smi_dict
                # save_file(smi_dict, model_id, save_dir, i, response.status_code)

            if response.status_code == 206 or response.status_code == 200:
                while (response.status_code == 206):
                    response = get("http://rest.ochem.eu/predict?MODELID={0}&SMILES={1}".format(model_id, url_smi))

                    # Pauses the loop between 1 - 2 seconds
                    sleep(randint(1, 2))

                    # If results are not ready, then continue
                    if response.text == 'not yet ready':
                        print('ochem api results --> not yet ready')
                        continue

                    # If error in results, then break
                    if response.status_code in error_codes:
                        break

                if response.status_code == 200:
                    err_code = None
                else:
                    err_code = 'ERROR'

                smi_dict[smiles].update({'results': json.loads(response.text),
                                         'response_code': int(response.status_code),
                                         'time_taken': round((time() - s_time), 3),
                                         'model_id': model_id, 'short_error': err_code})
                return smi_dict

        except Exception as e:
            smi_dict[smiles].update({'short_error': str(e.__class__.__name__),
                                     'long_error': str(e),
                                     'time_taken': round((time() - s_time), 3)})
            return smi_dict

#########<OCHEM ALOGPS CALCULATIONS [NOTE: can only be EXECUTED FROM VIA A LINUX MACHINE!]>#########
# USE DOCKER FOR BELOW TASK -->
class OchemToolALOGPS:
    def calculate_alogps(self, smi):

        cmd = ['./alogps-linux','--smiles', smi]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, error = p.communicate()
        return out, error


###############<FETCH PHYSICO CHEMICAL PROPERTIES>###########################
# Takes input as smiles
class FetchPhysicoProperty:

    # Get molecular weight of smiles string
    def get_molecular_wt(self, smi):

        try:
            m = Chem.MolFromSmiles(smi)
            return round(Descriptors.MolWt(m), 2)

        except:
            return '-'

    # Get molecular formula of smiles string
    def get_molecular_formula(self, smi):

        try:
            m = Chem.MolFromSmiles(smi)
            return CalcMolFormula(m)

        except:
            '-'
################################################################################

###############<FETCH ATTRIBUTES FROM CHEMICAL DATABASES, USING APIS / OTHER>###########################
# Takes input as smiles
class FetchChemoDB:

    # Convert to Canonical Smiles
    def get_canonical(self, smi):

        try:
            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol, True)
            return can_smi

        except:
            return None

    # Fetch Pubchem results
    def fetch_pubchem(self, smi):

        can_smi = self.get_canonical(smi)
        if can_smi == None:
            return '-', '-'

        try:
            # func_timeout runs for a certain time period. If results not returned in that time, it breaks
            # refer - https://pypi.org/project/func-timeout/
            r = func_timeout(pubchem_time_limit, get_compounds, args=(smi, 'smiles'))
            # r = get_compounds(smi, 'smiles')
            _cid = r[0].cid
            return 'https://pubchem.ncbi.nlm.nih.gov/compound/' + str(_cid), _cid

        except:
            return '-', '-'

    # Fetch DrugCentral results
    def fetch_drug_central(self, smi, _input):

        can_smi = self.get_canonical(smi)
        if can_smi == None:
            return '-', '-'

        # Read csv
        #df = pd.read_csv('drug_central_drugs.csv')
        df = pd.read_csv('drug_central_drugs-stand.csv')
        ### added by GK ###
        dc_dictn = dict(zip(df.ID, df.INN_cleaned))
        dc_dictn_inn = dict(zip(df.INN_cleaned, df.Canonical_Smiles))
        ##################-----------################

        try:
            # Check if query canonical smi matches with canonical smi in drugCentral db
            dc_id = df[df.Canonical_Smiles == can_smi]['ID'].values[0]
            dc_name = dc_dictn[dc_id] # added by gK
            dc_smiles_stand = dc_dictn_inn[dc_name]
            return 'http://drugcentral.org/drugcard/' + str(dc_id), dc_id, dc_name, dc_smiles_stand # dc_name added by gk

        except:
            try:
                # Convert to string
                _input = str(_input)

                # Convert to lowercase
                _input = _input.lower()

                # Remove leading and trailing spaces
                _input = _input.strip()

                # Matching query drug_name with that present in drugCentral db
                dc_id = df[df.INN_cleaned == _input]['ID'].values[0]
                dc_name = dc_dictn[dc_id] # added by Gk
                dc_smiles_stand = dc_dictn_inn[dc_name]
                return 'http://drugcentral.org/drugcard/' + str(dc_id), dc_id, dc_name, dc_smiles_stand # dc_name, dc_smiles_stand added by gk

            except:
                return '-', '-', '-', '-' # added '-' by gk
################################################################################

###############<CHECK INPUT TYPE>###########################
class CheckInput:

    # Convert to Canonical Smiles
    def get_canonical(self, smi):

        try:
            if len(smi) == 0:
                return None

            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol, True)
            return can_smi

        except:
            return None
    

    def check_input(self, _input):

        smi_flag = False
        drug_name_flag = False
        pubchem_cid_flag = False

        # First, check if canonical
        can_smi = self.get_canonical(_input)

        if can_smi != None:
            smi_flag = True
            try:
                drug_name = func_timeout(pubchem_time_limit, get_synonyms, args=(can_smi, 'smiles'))

                if not drug_name:
                    drug_name = '-'
                elif len(drug_name[0]['Synonym']) == 1:
                    drug_name = str(drug_name[0]['Synonym'][0])
                else:
                    drug_name = drug_name[0]['Synonym']
                    drug_name.sort(key=len)
                    drug_name = str(' | '.join(drug_name[0:2]))
            except:
                drug_name = '-'
            return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

        else:

            # Convert to string
            _input = str(_input)

            # Convert to lowercase
            _input = _input.lower()

            # Remove leading and trailing spaces
            _input = _input.strip()

            ######<CHECK IF PUBCHEM CID>######
            try:
                # Check if it is a PubChem CID
               
                r = func_timeout(pubchem_time_limit, get_compounds, args=(_input, 'cid'))

                # Get canonical smiles
                can_smi = r[0].canonical_smiles
                print(can_smi)
                can_smi = self.get_canonical(can_smi)

                if can_smi != None:
                    pubchem_cid_flag = True

                    try:
                        drug_name = func_timeout(pubchem_time_limit, get_synonyms, args=(can_smi, 'smiles'))
                        if len(drug_name[0]['Synonym']) == 1:
                            drug_name = str(drug_name[0]['Synonym'][0])
                        else:
                            drug_name = drug_name[0]['Synonym']
                            drug_name.sort(key=len)
                            drug_name = str(' | '.join(drug_name[0:2]))
                    except:
                        drug_name = '-'
                    return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

                else:
                    smi_flag = False
                    pubchem_cid_flag = False
                    drug_name = '-'

            except:
                smi_flag = False
                pubchem_cid_flag = False
                can_smi = None
                drug_name = '-'

            ######<CHECK IF DRUG NAME>######
            try:

                # Remove multiple spaces from between words
                _input = " ".join(_input.split())

                # Check if name present in pubchem
                r = func_timeout(pubchem_time_limit, get_compounds, args=(_input, 'name'))

                # Get canonical smiles
                can_smi = r[0].canonical_smiles
                can_smi = self.get_canonical(can_smi)

                if can_smi != None:
                    drug_name_flag = True

                    try:
                        drug_name = func_timeout(pubchem_time_limit, get_synonyms, args=(can_smi, 'smiles'))
                        if len(drug_name[0]['Synonym']) == 1:
                            drug_name = str(drug_name[0]['Synonym'][0])
                        else:
                            drug_name = drug_name[0]['Synonym']
                            drug_name.sort(key=len)
                            drug_name = str(' | '.join(drug_name[0:2]))
                    except:
                        drug_name = '-'
                    return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

                else:
                    smi_flag = False
                    drug_name_flag = False
                    drug_name = '-'
                    return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

            except:
                smi_flag = False
                drug_name_flag = False
                can_smi = None
                drug_name = '-'
                return can_smi, drug_name, smi_flag, drug_name_flag, pubchem_cid_flag

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

