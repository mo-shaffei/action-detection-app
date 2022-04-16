from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import threading
import logic
import pandas as pd
import pymongo
from pymongo import MongoClient

app = Flask(__name__)

mongo_Client = pymongo.MongoClient('localhost', 27017)
db = mongo_Client.webapp
results_data = db.results

@app.route('/')
def homepage():
    #results_data.remove()
    return render_template('homepage.html')


@app.route('/logs')
def logs():
    path = "video/"
    results = results_data.find()
    # columns = {"start", "end", "action", "confidence", "clip"}
    #df = pd.read_csv(path + "results.csv")  # load results csv file to dataframe
    # multiply confidence by 100 to get percentage then round to get integer
    #df.confidence = df.confidence * 100
    #df.confidence = df.confidence.round()
    # return response containing the dataframe
    #return render_template('logs.html', columns=df.columns, rows=df.to_dict('records'))
    return render_template('logs.html', results=results)


@app.route('/connect')
def connect():
    t = threading.Thread(target=logic.connect_thread)  # create new thread
    t.setDaemon(True)
    t.start()  # start thread
    return Response(status=204)


if __name__ == '__main__':
    app.run(debug=True)