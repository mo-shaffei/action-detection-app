from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import threading
import logic
import pandas as pd
import pymongo
from pymongo import MongoClient
import json

with open('config.json') as f:
    config = json.load(f)

app = Flask(__name__)
app.config.update(config)
mongo_Client = pymongo.MongoClient('localhost', 27017)
db = mongo_Client.webapp
results_data = db.results


@app.route('/')
def homepage():
    results_data.remove({})
    return render_template('homepage.html')


@app.route('/logs')
def logs():
    results = results_data.find()  # getting all results stored in the database
    return render_template('logs.html', results=results)


@app.route('/connect')
def connect():
    t = threading.Thread(target=logic.connect_thread, args=[app])  # create new thread
    t.setDaemon(True)
    t.start()  # start thread
    return Response(status=204)


if __name__ == '__main__':
    app.run(debug=True)
