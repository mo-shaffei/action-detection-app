from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import threading
import logic
import pandas as pd
import pymongo


app = Flask(__name__)

client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system

@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/logs')
def logs():
    path = "video/"
    df = pd.read_csv(path + "results.csv")  # load results csv file to dataframe
    # multiply confidence by 100 to get percentage then round to get integer
    df.confidence = df.confidence * 100
    df.confidence = df.confidence.round()
    # return response containing the dataframe
    return render_template('logs.html', columns=df.columns, rows=df.to_dict('records'))


@app.route('/connect')
def connect():
    t = threading.Thread(target=logic.connect_thread)  # create new thread
    t.setDaemon(True)
    t.start()  # start thread
    return Response(status=204)


if __name__ == '__main__':
    app.run(debug=True)
