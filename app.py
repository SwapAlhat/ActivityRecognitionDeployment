# Importing essential libraries
from flask import Flask, render_template, request
import os
import numpy as np
from wsgiref import simple_server
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import pickle
import webbrowser
import statsmodels.api as sm

# Load the Linear Regression model
filename = 'AtivityRecognition-LogisticRegresssion-lbfs-model.pkl'
lr = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        avg_rss12 = float(request.form['avg_rss12'])
        var_rss12 = float(request.form['var_rss12'])
        avg_rss13 = float(request.form['avg_rss13'])
        var_rss13 = float(request.form['var_rss13'])
        avg_rss23 = float(request.form['avg_rss23'])
        var_rss23 = float(request.form['var_rss23'])

        data = np.array([[avg_rss12, var_rss12, avg_rss13, var_rss13, avg_rss23,var_rss23]])
        my_prediction = lr.predict(data)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    port = int(os.getenv("PORT"))
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host,port=port, app=app)
    httpd.serve_forever()