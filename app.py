import threading
import logic
import pandas as pd
import pymongo
from pymongo import MongoClient
from datetime import datetime
import json
import uuid
from bson.json_util import dumps
import visualize
from flask import Flask, render_template, Response, request, redirect, url_for, flash, session

with open('config.json') as f:
    config = json.load(f)

app = Flask(__name__)
app.config.update(config)
mongo_Client = pymongo.MongoClient('localhost', 27017)

db = mongo_Client.webapp
results_data = db.results
# secret key is needed for session
app.secret_key = 'detectionappdljsaklqk24e21cjn!Ew@@dsa5'


@app.route("/", methods=["POST", "GET"])
def login():
    # results_data.remove({})
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("pass")

        if (username == "Admin") & (password == "123"):

            # Start the inference
            t = threading.Thread(target=logic.inference_thread, args=[app, [5555], True])  # create new thread
            t.setDaemon(True)
            t.start()  # start thread

            return redirect(url_for("logs"))
        else:
            flash("Incorrect Username or Password!")

    return render_template('login.html')


@app.route('/logs/')
def logs():
    n = 50
    results = results_data.find().sort("start", -1).limit(n)
    return render_template('logs.html', results=results, raw_results=results)


@app.route('/filtering/', methods=["GET", "POST"])
def filtering():
    # Retrieve the desired filters from the user
    confidence = request.form.get("confidence")
    start_date = request.form.get("start_date")
    start_time = request.form.get("start_time")
    end_date = request.form.get("end_date")
    end_time = request.form.get("end_time")

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

    # print("output\n action:{}\n conf:{}\n loc:{}\n cam:{}\n start:{}\n end:{}\n".format(action,
    #                                                                                    confidence, location, camera,
    #                                                                                    len(start), len(end)))

    # keys = ["action", "confidence", "clip", "location", "camera",
    #        "start_date", "start_time", "end_date", "end_time"]

    all_filters = {
        'action': action,
        'confidence': confidence,
        'location': location,
        'camera_id': camera,
        'start': start_date + ', ' + start_time,
        'end': end_date + ', ' + end_time
    }

    print("ALL FILTERS:::\n")
    print(all_filters)

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
            filters[key] = {'$gte': datetime.strptime(value, '%Y-%m-%d, %H:%M:%S')}

        #### end
        if key == 'end':
            print("------end filter-----")
            # store the value of "end" to be less than or equal to the input value in the filters dict
            filters[key] = {'$lte': datetime.strptime(value, '%Y-%m-%d, %H:%M:%S')}

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

    # storing the filters to be used in the sorting function, to apply sorting on filtered data
    session["filters"] = filters
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
            [(sorting, pymongo.ASCENDING)])  # sorting on filtered data Ascendingly,

    # remove the stored filters, so that if the user goes to the sorting option directly without filtering,
    # it doesn't use the last stored filters, instead, it sorts the whole results
    session["filters"] = {}
    return render_template('logs.html', results=results, raw_results=results_data)


# Implementing visualizations
@app.route('/callback', methods=['POST', 'GET'])
def cb():
    [g1, _, _, _] = visualize.plot_all(results_data, action=request.args.get('data'))
    return g1

    # return visualize.plots(results_data, request.args.get('data'))[0]


@app.route('/visualize/', methods=['GET', 'POST'])
def index():
    # [g1, g2, g3, g4, s1, s2, s3, s4] = visualize.plots(results_data, action='eating')

    [g1, g2, g3, g4] = visualize.plot_all(results_data, action='Eating')
    [s1, s2, s3, s4] = visualize.statistics_all(results_data)
    return render_template('visualize.html', graphJSON=g1, graph2JSON=g2,
                           graph3JSON=g3, graph4JSON=g4, top_action=s1, top_location=s2, top_camera=s3, min_camera=s4,
                           raw_results=results_data)


@app.route('/filter_visualizations/', methods=["GET", "POST"])
def filter_visualizations():
    # Retrieve the desired filters from the user
    confidence = request.form.get("confidence")
    start_date = request.form.get("start_date")
    start_time = request.form.get("start_time")
    end_date = request.form.get("end_date")
    end_time = request.form.get("end_time")

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

    # print("output\n action:{}\n conf:{}\n loc:{}\n cam:{}\n start:{}\n end:{}\n".format(action,
    #                                                                                    confidence, location, camera,
    #                                                                                    len(start), len(end)))

    # keys = ["action", "confidence", "clip", "location", "camera",
    #        "start_date", "start_time", "end_date", "end_time"]
    print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEE-----------{},{},{},{}.".format(start_date, start_time, end_date, end_time))

    all_filters = {
        'action': action,
        'confidence': confidence,
        'location': location,
        'camera_id': camera,
        'start': start_date + ', ' + start_time,
        'end': end_date + ', ' + end_time
    }

    print("ALL FILTERS:::\n")
    print(all_filters)

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
            filters[key] = {'$gte': datetime.strptime(value, '%Y-%m-%d, %H:%M:%S')}

        #### end
        if key == 'end':
            print("------end filter-----")
            # store the value of "end" to be less than or equal to the input value in the filters dict
            filters[key] = {'$lte': datetime.strptime(value, '%Y-%m-%d, %H:%M:%S')}

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

    # [g1, g2, g3, g4, s1, s2, s3, s4] = visualize.plots(results, action='eating')
    [g1, g2, g3, g4] = visualize.plot_all(results, action='Eating')
    [s1, s2, s3, s4] = visualize.statistics_all(results)

    return render_template('visualize.html', graphJSON=g1, graph2JSON=g2,
                           graph3JSON=g3, graph4JSON=g4, top_action=s1, top_location=s2, top_camera=s3, min_camera=s4,
                           raw_results=results_data)


if __name__ == '__main__':
    app.run(debug=True)
