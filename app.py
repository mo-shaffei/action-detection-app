from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify, session
import threading
import logic
import pandas as pd
import pymongo
from pymongo import MongoClient
import json
import uuid
from bson.json_util import dumps
import plotly
import plotly.express as px
from pandas import DataFrame
import plotly.graph_objects as go


with open('config.json') as f:
    config = json.load(f)

app = Flask(__name__)
app.config.update(config)
mongo_Client = pymongo.MongoClient('localhost', 27017)

db = mongo_Client.webapp
results_data = db.results
# secret key is needed for session
app.secret_key = 'detectionappdljsaklqk24e21cjn!Ew@@dsa5'


@app.route('/')
def homepage():
    results_data.remove({})
    return render_template('homepage.html')


@app.route('/connect/')
def connect():
    t = threading.Thread(target=logic.connect_thread, args=[app])  # create new thread
    t.setDaemon(True)
    t.start()  # start thread
    return Response(status=204)


@app.route('/logs/')
def logs():
    results_data.insert_one({"camera_id": '001', "start": 0, "end": 1,
                                 "action": "smoking", "confidence": 50,
                                 "clip": 2, "location": 'location'}) 
    results_data.insert_one({"camera_id": '002', "start": 5, "end": 6,
                                 "action": "eating", "confidence": 30,
                                 "clip": 5, "location": 'location'})  
    results_data.insert_one({"camera_id": '002', "start": 7, "end": 8,
                                 "action": "drinking", "confidence": 30,
                                 "clip": 9, "location": 'location2'}) 
    results_data.insert_one({"camera_id": '002', "start": 7, "end": 8,
                                 "action": "eating", "confidence": 30,
                                 "clip": 9, "location": 'location2'}) 
    results_data.insert_one({"camera_id": '002', "start": 8, "end": 9,
                                 "action": "drinking", "confidence": 40,
                                 "clip": 9, "location": 'location2'})
    results_data.insert_one({"camera_id": '002', "start": 8, "end": 9,
                                 "action": "eating", "confidence": 70,
                                 "clip": 10, "location": 'location2'}) 
    results_data.insert_one({"camera_id": '002', "start": 8, "end": 9,
                                 "action": "eating", "confidence": 80,
                                 "clip": 10, "location": 'location2'})                                                                                                                      
    n = 1000
    results = results_data.find({}).limit(n)  # getting results stored in the database (last n)
    return render_template('logs.html', results=results, raw_results=results)


@app.route('/filtering/', methods=["GET", "POST"])
def filtering():
    # Retrieve the desired filters from the user
    confidence = request.form.get("confidence")
    start = request.form.get("start")
    end = request.form.get("end")

    # For each of the action, location and camera_id, store them in a list if the user specified more than one value
    action = request.form.getlist("action")
    if len(action) == 1:
        action = request.form.get("action")

    location = request.form.getlist("location")
    if len(location) == 1:
        location = request.form.get("location")

    camera = request.form.getlist("camera_id")
    if len(camera) == 1:
        camera = request.form.get("camera_id")

    print("output\n action:{}\n conf:{}\n loc:{}\n cam:{}\n start:{}\n end:{}\n".format(action,
                                                                                        confidence, location, camera,
                                                                                        len(start), len(end)))

    # keys = ["action", "confidence", "clip", "location", "camera",
    #        "start_date", "start_time", "end_date", "end_time"]

    all_filters = {
        'action': action,
        'confidence': confidence,
        'location': location,
        'camera_id': camera,
        'start': start,
        'end': end,
    }

    filters = {}
    for key, value in all_filters.items():
        # If there's no filter on one of the columns, exclude it from the filters dictionary
        if value == 'All' or value == "" or value == ['All']:
            print("------No filter-----")
            # Jump to the next key,value
            continue

        #### start
        if key == 'start':
            print("------start filter-----")
            # store the value of "start" to be greater than or equal to the input value in the filters dict
            filters[key] = {'$gte': int(value)}

        #### end
        if key == 'end':
            print("------end filter-----")
            # store the value of "end" to be less than or equal to the input value in the filters dict
            filters[key] = {'$lte': int(value)}

        #### confidence
        if key == 'confidence':
            print("------confidence filter-----")
            value = int(value)
            # store the value of "confidence" to be greater than or equal to the input value in the filters dict
            filters[key] = {'$gte': value}
            print(filters[key])
            print(filters)

        # If the user specified more than one filter
        if type(value) == list:
            print("------ list filter-----")
            # store the value of "key" to be in the input list in the filters dict
            filters[key] = {'$in': value}
            print("Key: {}\n value: {}".format(key, value))

        #### camera_id
        elif key == 'camera_id':
            print("****------camera_id filter-----")
            print(key, value)
            # store the value of "camera_id" to be the uuid of the input value in the filters dict
            # filters[key] = uuid.UUID(value)
            filters[key] = value
            print(filters[key], type(filters[key]))

        #### action, location
        # If the user specified one action/location only
        elif key == 'action' or key == 'location':
            print("****------action/ loc filter-----")
            # store the value of "action"/ "location" to be the input value in the filters dict
            filters[key] = value

        # print("Key: {}\n value: {}".format(key, value))
    print("Filters dict: {}\n ".format(filters))

    results = results_data.find(filter=filters)
    print("results type is {}".format(type(results)))
    # print("results: {}".format(results))
    # return "output " + action + confidence + clip + location + camera + start_date + start_time + end_date + end_time
    # session["filtered_data"]=dumps(results_data.find(filter=filters))
    session[
        "filters"] = filters  # storing the filters to be used in the sorting function, to apply sorting on filtered data
    return render_template('logs.html', results=results, raw_results=results_data)


@app.route('/sorting/', methods=["GET", "POST"])
def sorting():
    """"
    
    """
    filters = session.get("filters", None)  # getting the filters from the filtering function
    sorting = request.form.get(
        "sorting")  # returns the option that the user want to sort by (ex: action, confidence, ...etc)
    sorting_order = request.form.get("sorting_order")  # returns ascending or descending
    if sorting_order == "Descending":
        results = results_data.find(filter=filters,
                                    sort=[(sorting, pymongo.DESCENDING)])  # sorting on filtered data Descendingly
    else:
        results = results_data.find(filter=filters).sort(
            [(sorting, pymongo.ASCENDING)])  # sorting on filtered data ASCENDINGly

    session[
        "filters"] = {}  # remove the stored filters, so that if the user goes to the sorting option directly without filtering, it doesn't use the last stored filters, instead, it sorts the whole results
    return render_template('logs.html', results=results, raw_results=results_data)

#Implementing visualizations
@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return gm(request.args.get('data'))
   
@app.route('/visualize/', methods=['POST', 'GET'])
def index():
    return render_template('visualize.html',  graphJSON=gm(), graph2JSON=g2(), graph3JSON=g3())

def gm(action='eating'):
    
    filtered_data = results_data.aggregate([
    {
        "$match" : {
            "action" : {
                "$eq" : action                               #filtering the entered action, then
            } 
        }
    },
    {
        "$group" : {"_id" : "$start", "count": {"$sum" : 1}}  #counting the rows of this action for each start time
    }
])

    list_data = list(filtered_data)
    df = DataFrame(list_data)
    df.head()

    fig = px.line(df, x="_id", y="count", labels={'x': 'start'}).update_layout(xaxis_title="start time")
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #print(fig.data[0])
    
    return graphJSON


def g2(): 

    #filtered_data = results_data.aggregate([
    #    {
    #        "$group" : {
    #            "_id" :  {  "location": "$location", "action": "$action"} ,
    #            "count": { "$sum" : 1 } }
    #    },

        #{
        #    "$group" : {"_id" : "$_id.location", "actions": { 
        #      "$push": { "action":"$_id.action", "count":"$count" }
        #} } 
        #}
    #])
    
    list_data = list(results_data.find())
    df = DataFrame(list_data)
    #print(df)
    #print(type(df._id))

    fig2 = px.bar(df, x="location", y="camera_id", color='action',
      barmode='group')

    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return graph2JSON


def g3():
    counted_actions = results_data.aggregate([
        {
            "$group" : {"_id" : "$action", "count": {"$sum" : 1}}  
        }
    ])

    list_data = list(counted_actions)
    df = DataFrame(list_data)

    fig3 = px.pie(df, values='count', names='_id')

    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    return graph3JSON


