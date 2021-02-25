#######################################################################
# http://sirimullaresearchgroup.com/                                  #
# University of Texas, El Paso, Tx, USA                               #
# Last modified: 29/08/2020                                           #
#######################################################################

from flask import Flask, jsonify, render_template, request, url_for, redirect, make_response
import os, sys, joblib, time, json
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from features import FeaturesGeneration
from pprint import pprint
from rdkit import Chem
import cairosvg
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import base64
import io
from PIL import Image
from rdkit.Chem import rdDepictor

app = Flask(__name__)

@app.route("/")
def home():

    return jsonify({'message': 'SERVER IS RUNNING'})

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method in ['GET', 'POST']:
        # request = add_header(request)
        _input = request.form['smiles']
        print("PREDICTING FOR {}".format(_input), file = sys.stderr)

        final_results = {}

        if _input == '' or _input == None or len(_input) == 0:
            final_results['error'] = {'message': "INPUT ERROR", 'error': "EMPTY INPUT"}
            final_results['error_flag'] = 'TRUE'
            print("ERROR: EMPTY INPUT")

            return jsonify(final_results)

        smiles = _input

        if smiles == None:
            final_results['error'] = {'message': "INPUT ERROR", 'error': "INVALID INPUT"}
            final_results['error_flag'] = 'TRUE'
            print("ERROR: INVALID INPUT")
            return jsonify(final_results)

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


        def get_features(smiles):
            fg = FeaturesGeneration()

            features = fg.get_fingerprints(smiles)
            return features

        def get_results(smiles):
            # Load models
            with open('models.txt', 'r') as f:
                models = f.read().splitlines()
                model_names = [model_path.split('_')[0] for model_path in models]
                models = [joblib.load(os.path.join('models', model_path)) for model_path in models]
            

            features = get_features(smiles)

            model_result = {}

            for model_name, model in tqdm(zip(model_names, models)):
                label_zero = model.predict_proba(features)[0][0].round(3)
                label_one = model.predict_proba(features)[0][1].round(3)

                if label_one >= 0.5:
                    model_result['Prediction'] = 'B-Arrestin'
                    model_result['Confidence'] = label_one
                    model_result['GPCR'] = gpcr_results(features)
                else:
                    model_result['Prediction'] = 'G-Protein'
                    model_result['Confidence'] = label_zero
                    model_result['GPCR'] = gpcr_results(features)

            final_results[smiles] = model_result
            
            final_results['smiles']=smiles
            final_results['predictions'] = model_result
            final_results['error_flag'] = 'FALSE'

            return final_results
        
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

        try:

            final_results = get_results(smiles)
            binary_image = smi_to_png(smiles, './static/images/', get_binary=True)
            final_results['image']=binary_image

            return jsonify(final_results)

        except Exception as e:
            final_results = dict()
            final_results['error'] = {'message': "SCRIPT ERROR", 'error': str(e)}
            final_results['error_flag'] = 'TRUE'
            return jsonify(final_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
