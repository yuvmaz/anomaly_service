#!/usr/bin/env python3

from flask import Flask, jsonify, request, abort
from pickle import load 
from scipy.stats import t
from keras.models import load_model
import numpy as np
import sys

model = load_model("logs.h5")
with open("logs.pkl", "rb") as f:
    info = load(f)

look_back = info['look_back']
previous_timesteps = []
error_rv = t(info['nu'], info['mu'],info['sigma'])


num_anomalous_timesteps = 0


app = Flask(__name__)


def predict_timestamp(previous_timesteps, actual, model, look_back, error_rv):
    assert previous_timesteps is not None, "previous_timesteps must be provided"
    assert isinstance(actual, float), "Actual must be a float value"
    assert model is not None, "Model must be provided"
    assert isinstance(look_back, int) and look_back > 0, "look_back must be and integer and > 0"
    assert error_rv is not None, "Error distribution must be provide"
    
    assert len(previous_timesteps) == look_back, "Length of timesteps must be equal to {}".format(look_back)
    
    prediction = model.predict(previous_timesteps.reshape((1,len(previous_timesteps),1))).flatten()[0]
    normal_prob = 1 - error_rv.cdf(actual - prediction)
    
    return prediction, normal_prob


@app.route("/report", methods=['POST'])
def report():
    try:
        global previous_timesteps, num_anomalous_timesteps

        if not request.json or not 'eventCount' in request.json:
            abort(400)

        event_count = request.json['eventCount']
        anomaly_threshold = 0.03
        if 'anomalyThreshold' in request.json:
            anomaly_threshold = float(request.json['anomalyThreshold'])

        if len(previous_timesteps) < look_back:
            msg = "More data needed, currently have {} items".format(len(previous_timesteps))
            code = 201
        else:
            code = 200
            predicted_event_count, prob_normal_behavior = \
                    predict_timestamp(np.log(previous_timesteps), np.log(event_count), model, look_back, \
                                        error_rv)
            if prob_normal_behavior < anomaly_threshold:
                num_anomalous_timesteps += 1
            else:
                num_anomalous_timesteps = 0


            msg = jsonify(
                           {
                                "predictedEventCount":  int(np.exp(predicted_event_count)), 
                                "probNormalBehavior": "{:.5f}".format(prob_normal_behavior),
                                "numAnomalousTimesteps": int(num_anomalous_timesteps),
                           }
                         )

        previous_timesteps.insert(0, event_count)
        previous_timesteps = previous_timesteps[0:look_back]

    except:
        msg = "Error: {}".format(sys.exc_info()[0])
        code = 404

    return msg, code
