#!/usr/bin/env python3

from flask import Flask, jsonify, request, abort
from pickle import load 
from scipy.stats import t
from keras.models import model_from_json
import numpy as np
import sys
import traceback


model_store = {}
lookback_store = {} 
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
    error_rv_store[model_name] = t(model_info['nu'], model_info['mu'],model_info['sigma'])
    num_anamolous_timesteps_store[model_name] = 0

print("Stores ready")


app = Flask(__name__)


def predict_timestamp(previous_timesteps, model_name, error_rv):

    model = model_store[model_name]
    error_rv = error_rv_store[model_name]

    assert previous_timesteps is not None, "previous_timesteps must be provided"
    assert model is not None, "Model must be provided"
    assert error_rv is not None, "Error distribution must be provide"

    try:
        prediction = model.predict(previous_timesteps.reshape((1,len(previous_timesteps),1))).flatten()[0]
        normal_prob = 1 - error_rv.cdf(previous_timesteps[-1] - prediction)
    except Exception as e:
        print(e)
    
    return prediction, normal_prob

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify(list(model_store.keys()))

def handle_single_report(json_object):
    msg, code = '', 200

    if not 'count' in json_object or not 'name' in json_object:
        return ('Missing count or name keys', 400)

    id_value = None
    if 'id' in json_object:
        id_value = json_object['id']

    event_counts = json_object['count']
    if type(event_counts) == float or type(event_counts) == int:
        event_counts = [event_counts]

    anomaly_threshold = 0.03
    if 'anomalyThreshold' in json_object:
        anomaly_threshold = float(json_object['anomalyThreshold'])

    model_name = json_object['name']


    if model_name not in model_store.keys():
        msg = { 
                "msg": "Unknown model {}".format(model_name)
              }
        code = 400

    else:
        look_back = lookback_store[model_name]
        error_rv = error_rv_store[model_name]

        if len(event_counts) < look_back:
            msg = {
                        "msg": "Not enough data, at least {} counts required".format(look_back)
                  }
            code = 400 


        else:
            code = 200
            previous_timesteps = np.array(event_counts[-look_back:], dtype=float)
            predicted_event_count, prob_normal_behavior = \
                    predict_timestamp(np.log(previous_timesteps), model_name, error_rv)

            if prob_normal_behavior < anomaly_threshold:
                num_anamolous_timesteps_store[model_name] += 1
            else:
                num_anamolous_timesteps_store[model_name] = 0

            msg = {
                    "predictedEventCount":  int(np.exp(predicted_event_count)), 
                    "probNormalBehavior": "{:.5f}".format(prob_normal_behavior),
                    "numAnomalousTimesteps": int(num_anamolous_timesteps_store[model_name]),
                   }

    if id_value:
        msg['id'] = id_value
    msg['name'] = model_name

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

