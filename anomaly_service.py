#!/usr/bin/env python3

from flask import Flask, jsonify, request, abort
from pickle import load 
from scipy.stats import t
from keras.models import model_from_json
import numpy as np
import sys


model_store = {}
lookback_store = {} 
prev_timestep_store = {}
error_rv_store = {}
num_anamolous_timesteps_store = {}

with open('all_models.pkl', 'rb') as f:
    models_repr = load(f)


for model_name in models_repr:
    print("Processing model {}...".format(model_name))

    this_model_repr = models_repr[model_name]
    model = model_from_json(this_model_repr[0])
    model.set_weights(this_model_repr[1])

    model_store[model_name] = model
    model_info = this_model_repr[2]
    lookback_store[model_name] = model_info['look_back']
    prev_timestep_store[model_name] = []
    error_rv_store[model_name] = t(model_info['nu'], model_info['mu'],model_info['sigma'])
    num_anamolous_timesteps_store[model_name] = 0

print("Stores ready")


app = Flask(__name__)


def predict_timestamp(previous_timesteps, actual, model_name, look_back, error_rv):

    model = model_store[model_name]
    previous_timesteps = np.array(prev_timestep_store[model_name], dtype=float)
    error_rv = error_rv_store[model_name]

    assert previous_timesteps is not None, "previous_timesteps must be provided"
    assert isinstance(actual, float), "Actual must be a float value"
    assert model is not None, "Model must be provided"
    assert isinstance(look_back, int) and look_back > 0, "look_back must be and integer and > 0"
    assert error_rv is not None, "Error distribution must be provide"

    
    assert len(previous_timesteps) == look_back, "Length of timesteps must be equal to {}".format(look_back)

    try:
        prediction = model.predict(previous_timesteps.reshape((1,len(previous_timesteps),1))).flatten()[0]
        normal_prob = 1 - error_rv.cdf(actual - prediction)
    except Exception as e:
        print(e)
    
    return prediction, normal_prob

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify(list(model_store.keys()))

def handle_single_report(json_object):
    msg, code = '', 200

    if not 'eventCount' in json_object or not 'modelName' in json_object:
        return ('Missing eventCount or modelName keys', 400)

    id_value = None
    if 'id' in json_object:
        id_value = json_object['id']

    event_count = json_object['eventCount']
    anomaly_threshold = 0.03
    if 'anomalyThreshold' in json_object:
        anomaly_threshold = float(json_object['anomalyThreshold'])
    model_name = json_object['modelName']


    if model_name not in model_store.keys():
        msg = { 
                "msg": "Unknown model {}".format(model_name)
              }
        code = 400

    else:
        previous_timesteps = prev_timestep_store[model_name]
        look_back = lookback_store[model_name]
        error_rv = error_rv_store[model_name]

        if len(previous_timesteps) < look_back:
            msg = {
                        "msg": "More data needed, currently have {} items".format(len(previous_timesteps))
                  }
            code = 201


        else:
            code = 200

            predicted_event_count, prob_normal_behavior = \
                    predict_timestamp(np.log(previous_timesteps), np.log(event_count), model_name, look_back, \
                                        error_rv)

            if prob_normal_behavior < anomaly_threshold:
                num_anamolous_timesteps_store[model_name] += 1
            else:
                num_anamolous_timesteps_store[model_name] = 0

            msg = {
                    "predictedEventCount":  int(np.exp(predicted_event_count)), 
                    "probNormalBehavior": "{:.5f}".format(prob_normal_behavior),
                    "numAnomalousTimesteps": int(num_anamolous_timesteps_store[model_name]),
                   }

        prev_timestep_store[model_name].insert(0, event_count)
        prev_timestep_store[model_name] = prev_timestep_store[model_name][0:look_back]

    if id_value:
        msg['id'] = id_value
    return msg, code



MSG = 0
CODE = 1

@app.route("/report", methods=['POST'])
def report():
    try:
        calls = None 
        responses = []
        
        if type(request.json) is dict:
            calls = [request.json]
        elif type(request.json) is list:
            calls = request.json
        else:
            return "Cannot handle data format of type {}".format(type(request.json)), 400

        for this_call in calls:
            msg, code = handle_single_report(this_call)
            responses.append((msg, code))

        if len(responses) == 1:
            single_response = responses[0]
            return single_response[MSG], single_response[CODE]
        else:
            return jsonify(responses), 200

    except:
        return "Error: {}".format(sys.exc_info()[0]), 400 

