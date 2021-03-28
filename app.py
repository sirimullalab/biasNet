#######################################################################
# http://sirimullaresearchgroup.com/                                  #
# University of Texas, El Paso, Tx, USA                               #
# Last modified: 29/08/2020                                           #
#######################################################################

from flask import Flask, jsonify, render_template, request, url_for, redirect, make_response
import cairosvg
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import base64
import io
from PIL import Image
from run_biasnet import preprocess_smi, OchemToolALOGPS, FetchPhysicoProperty, FetchChemoDB, CheckInput, OchemAPIResults
import sys
from rdkit.Chem import rdDepictor
import json
import pickle
from  collections import OrderedDict
import pprint
import joblib,os
from features import FeaturesGeneration
from tqdm import tqdm

USE_OCHEM_API = False # If True, ochem API will be used for ALOGPS calculations (instead of ochem Tool)

app = Flask(__name__)


@app.route("/")
def home():

    return jsonify({'message': 'SERVER IS RUNNING'})

@app.route("/predict", methods=['GET', 'POST'])
def predict():

    if request.method in ['GET', 'POST']:
        # request = add_header(request)
        _input = request.form['smiles']
        
        print("PREDICTING FOR {}".format(_input), file=sys.stderr)
        
        # smiles = 'CCCCCC'
        all_dict = {}
        
        if _input == '' or _input == None or len(_input) == 0:
            all_dict['error'] = {'message': "INPUT ERROR", 'error': "EMPTY INPUT"}
            all_dict['error_flag'] = 'TRUE'
            print("ERROR: EMPTY INPUT")
            return jsonify(all_dict)#jsonify({'message': "SMILES ERROR", 'error': "EMPTY SMILES"})

        #Check input type
        smiles, synonyms, smi_flag, drug_name_flag, pubchem_cid_flag = CheckInput().check_input(_input)
        # Preprocessing the smiles
        processed_smiles = preprocess_smi(smiles)
        
        if processed_smiles == None:
            all_dict['error'] = {'message': "INPUT ERROR", 'error': "INVALID INPUT"}
            all_dict['error_flag'] = 'TRUE'
            print("ERROR: INVALID INPUT")
            return jsonify(all_dict)#jsonify({'message': "SMILES ERROR", 'error': "EMPTY SMILES"})

        

        try:
            # Below, getting attributes for using processed smiles -->
            with open('lookup_table_smiles.json') as smiles_file:
                lookup_table = json.load(smiles_file)

            def gpcr_results(features):
                with open('gpcr.txt', 'r') as f:
                    models = f.read().splitlines()
                    model_names = [model_path.split('_')[0] for model_path in models]
                    models = [joblib.load(os.path.join('gpcr', model_path)) for model_path in models]

                gpcr_model_result = {}

                for model_name, model in tqdm(zip(model_names, models)):
                    label_zero = model.predict_proba(features)[0][0].round(3)
                    label_one = model.predict_proba(features)[0][1].round(3)

                    if label_one >= 0.5:
                        gpcr_model_result['GPCR_Prediction'] = 'Non-GPCR'
                        gpcr_model_result['GPCR_Confidence'] = label_one                   
                    else:
                        gpcr_model_result['GPCR_Prediction'] = 'GPCR'
                        gpcr_model_result['GPCR_Confidence'] = label_zero                  
                return gpcr_model_result

            def get_results(smiles):
                # Load models
                with open('models.txt', 'r') as f:
                    models = f.read().splitlines()
                    model_names = [model_path.split('_')[0] for model_path in models]
                    models = [joblib.load(os.path.join('models', model_path)) for model_path in models]
            

                features = get_features(smiles)

                model_result = {}
                gpcr ={}
                for model_name, model in tqdm(zip(model_names, models)):
                    label_zero = model.predict_proba(features)[0][0].round(3)
                    label_one = model.predict_proba(features)[0][1].round(3)

                    if label_one >= 0.5:
                        model_result['Prediction'] = 'B-Arrestin'
                        model_result['Confidence'] = label_one
                        gpcr = gpcr_results(features)
                        model_result['GPCR_Prediction'] = gpcr['GPCR_Prediction']
                        model_result['GPCR_Confidence'] = gpcr['GPCR_Confidence']

                    else:
                        model_result['Prediction'] = 'G-Protein'
                        model_result['Confidence'] = label_zero
                        gpcr = gpcr_results(features)
                        model_result['GPCR_Prediction'] = gpcr['GPCR_Prediction']
                        model_result['GPCR_Confidence'] = gpcr['GPCR_Confidence']

                all_dict[smiles] = model_result
            
                all_dict['smiles']=smiles
                all_dict['predictions'] = model_result
                all_dict['error_flag'] = 'FALSE'

                return all_dict
            
            def get_features(processed_smiles):
                fg = FeaturesGeneration()
                features = fg.get_fingerprints(processed_smiles)
                return features

            

            def check_lookup_table(processed_smiles):
                lookup_table_result = {}
                if lookup_table[processed_smiles] == 1:
                    lookup_table_result['Prediction'] = 'B-Arrestin'
                    lookup_table_result['Confidence'] = '1.0'
                    lookup_table_result['GPCR_Prediction'] = 'GPCR'
                    lookup_table_result['GPCR_Confidence'] ='1.0'
                else:
                    lookup_table_result['Prediction'] = 'G-Protien'
                    lookup_table_result['Confidence'] = '1.0'
                    lookup_table_result['GPCR_Prediction'] = 'GPCR'
                    lookup_table_result['GPCR_Confidence'] ='1.0'
                all_dict[smiles]=lookup_table_result
                all_dict['predictions'] = lookup_table_result
                return all_dict

            def check_conditions_of_finalresults(prediction_results):
                bias_prediction = prediction_results['predictions']
                if bias_prediction['GPCR_Prediction'] == 'Non-GPCR':
                    bias_prediction['Prediction'] = '-'
                    bias_prediction['Confidence'] = 'Out of AD'
                    bias_prediction['GPCR_Prediction'] = bias_prediction['GPCR_Prediction']
                    bias_prediction['GPCR_Confidence'] = bias_prediction['GPCR_Confidence']
                all_dict[smiles] = bias_prediction
                all_dict['predictions'] = bias_prediction
                return all_dict
            
            if smiles in lookup_table:
                all_dict = check_lookup_table(smiles)
            else:
                all_dict = get_results(smiles)
                all_dict=check_conditions_of_finalresults(all_dict)
            
            molecular_wt = FetchPhysicoProperty().get_molecular_wt(processed_smiles)
            molecular_formula = FetchPhysicoProperty().get_molecular_formula(processed_smiles)
            # Below, getting attributes for using query smiles -->
            pubchem_link, pubchem_cid = FetchChemoDB().fetch_pubchem(smiles)
            # This line modified by GK
            drug_central_link, drug_central_id, drug_central_name, dc_smiles_stand = FetchChemoDB().fetch_drug_central(smiles, _input)

           
            

            

            if not USE_OCHEM_API:
            #########<OCHEM ALOGPS TOOL CALCULATIONS [NOTE: can only be EXECUTED FROM VIA A LINUX MACHINE!]>#########
            # USE DOCKER FOR BELOW TASK -->
                try:
                    o = OchemToolALOGPS()
                    out, err = o.calculate_alogps(processed_smiles)
                    out, err = str(out), str(err)
                    if 'error' in out or out == '':
                        logp = 'smi_error'
                        logs = 'smi_error'
                    else:
                        s = out
                        logp = s[s.find('logP:') + len('logP:'):s.find('(', s.find('logP:') + len('logP:'))]
                        logs = s[s.find('logS:') + len('logS:'):s.find('(', s.find('logS:') + len('logS:'))]

                except:
                    logp = 'script_error'
                    logs = 'script_error'
            ######################################
            if USE_OCHEM_API:
            #########<OCHEM ALOGPS API CALCULATIONS>#########
                try:
                    ochem_api_ob = OchemAPIResults()
                    logp = ochem_api_ob.get_ochem_model_results(processed_smiles, 535) # logp
                    logs = ochem_api_ob.get_ochem_model_results(processed_smiles, 536)  # logs

                except:
                    logp = '-'
                    logs = '-'
            ######################################

            

            def smi_to_png(smi, query_smi_path, get_binary=False):

                def moltosvg(mol, molSize=(300, 300), kekulize=True):
                    mc = Chem.Mol(mol.ToBinary())
                    if kekulize:
                        try:
                            Chem.Kekulize(mc)
                        except:
                            mc = Chem.Mol(mol.ToBinary())
                    if not mc.GetNumConformers():
                        rdDepictor.Compute2DCoords(mc)
                    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
                    drawer.DrawMolecule(mc)
                    drawer.FinishDrawing()
                    svg = drawer.GetDrawingText()
                    return svg

                mol = Chem.MolFromSmiles(smi)
                svg_vector = moltosvg(mol)
                cairosvg.svg2png(bytestring=svg_vector, write_to=query_smi_path + 'query_smi.png')
                img = Image.open(query_smi_path + 'query_smi.png', mode='r')
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
                return encoded_img

            binary_image = smi_to_png(processed_smiles, './static/images/', get_binary=True)

            ## added by GK ###
            #synonyms = synonyms.split('|')
            #print('Synonyms ', synonyms)
            #synonyms.sort(key=len)
            #print('Leng of sy', len(synonyms))
            #synonyms = '|'.join(synonyms)
            #print(synonyms)
            if drug_central_name!='-':
                syno_0 = synonyms.split('|')[0].lower()
                syno_0 = syno_0.strip()
                syno_1 = synonyms.split('|')[1].lower()
                syno_1 = syno_1.strip()

                if syno_0==drug_central_name:
                    synonyms_modified = drug_central_name+' | '+synonyms.split('|')[1]
                else:
                    synonyms_modified = drug_central_name+' | '+synonyms.split('|')[0]
            else:

                synonyms_modified = synonyms.split('|')[0].lower()+' | '+ synonyms.split('|')[1].lower()
                pass
            ## Check if dc has smiles
            if dc_smiles_stand !='-':
                all_dict['processed_query_smiles'] = dc_smiles_stand 
            else:
                all_dict['processed_query_smiles'] = processed_smiles
                pass
            ############## ###########3
            print(smiles)
            all_dict['image'] = binary_image
            all_dict['logp'] = logp
            all_dict['logs'] = logs
            all_dict['molecular_wt'] = molecular_wt
            all_dict['molecular_formula'] = molecular_formula
            all_dict['pubchem_link'] = pubchem_link
            all_dict['pubchem_cid'] = str(pubchem_cid)
            all_dict['drug_central_link'] = drug_central_link
            all_dict['drug_central_id'] = str(drug_central_id)
            all_dict['synonyms'] = synonyms_modified

            

        except Exception as e:
            all_dict['error'] = {'message': "SCRIPT ERROR", 'error': str(e)}
            all_dict['error_flag'] = 'TRUE'
            return jsonify(all_dict)
        return jsonify(all_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run(debug=True)
