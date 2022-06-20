import plotly
import plotly.express as px
from pandas import DataFrame
import plotly.graph_objects as go
import json
from bson.json_util import dumps
import pymongo


def helper(results):
    # Handling cursor problem for filtered data
    coll = False
    if type(results) == pymongo.cursor.Cursor:
        print("ENTERED HEREEEEEEEE\n\n\n")
        client = pymongo.MongoClient()
        db = client["data"]
        coll = db["DATA"]

        list_dict = []

        for i in results:
            list_dict.append(i.copy())

        for dictionary in list_dict:
            coll.insert_one(dictionary)

    if coll:
        results = coll

    print("resultssssssss type is {}".format(type(results)))
    filtered_data = results.aggregate([  # filtered data for g2 and g4
        {
            "$group": {
                "_id": {"location": "$location", "action": "$action"},
                "count": {"$sum": 1}}
        },

        {
            "$group": {"_id": "$_id.location", "actions": {
                "$push": {"action": "$_id.action", "count": "$count"}
            }}
        }
    ])

    # filtered_data = filtered_data.find()
    list_data = list(filtered_data)
    df = DataFrame(list_data)
    x = [];
    y = [];
    z = [];
    i = -1
    for location in df._id:
        i = i + 1
        for item in df.actions[i]:
            x.append(item.pop('action'))
            y.append(item.pop('count'))
            z.append(location)

        new_df = DataFrame(list(zip(x, y, z)),  # dataframe used in g2 and g4
                           columns=['action', 'count', 'location'])
    ############################################################################

    counted_actions = results.aggregate([  # counted actions needed in g3 and s1
        {
            "$group": {"_id": "$action", "count": {"$sum": 1}}
        }
    ])
    list_counted_data = list(counted_actions)
    ############################################################################

    return filtered_data, new_df, list_counted_data, x, y, z


def g1(results, action):  # plots graph 1 (g1)

    # Handling cursor problem for filtered data
    coll = False
    if type(results) == pymongo.cursor.Cursor:
        print("ENTERED HEREEEEEEEE\n\n\n")
        client = pymongo.MongoClient()
        db = client["data"]
        coll = db["DATA"]

        list_dict = []

        for i in results:
            list_dict.append(i.copy())

        for dictionary in list_dict:
            coll.insert_one(dictionary)

    if coll:
        results = coll

    filtered_data = results.aggregate([
        {
            "$match": {
                "action": {
                    "$eq": action  # filtering the entered action, then
                }
            }
        },
        {
            "$group": {"_id": "$start", "count": {"$sum": 1}}  # counting the rows of this action for each start time
        }
    ])

    list_data = list(filtered_data)
    df = DataFrame(list_data)

    fig = px.bar(df, x="_id", y="count", labels={'x': 'start'}).update_layout(xaxis_title="start time")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def g2(new_df):  # plots graph 2 (g2)

    fig2 = px.bar(new_df, x="location", y="count", color='action',  # a bar plot of locations vs count of each action
                  barmode='group')

    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return graph2JSON


def g3(list_counted_data):  # plots graph 3 (g3)

    df = DataFrame(list_counted_data)

    fig3 = px.pie(df, values='count', names='_id')  # pie chart of count of all actions

    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    return graph3JSON


def g4(x, y, z):  # plots graph 4 (g4)

    fig4 = go.Figure(data=go.Heatmap({'z': y, 'x': x, 'y': z}))  # a heatmap of actions vs locations

    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    return graph4JSON


###########################################################################################

def s1(df_original):  # return statistics 1 (Top action)
    top_action = df_original['action'].value_counts().idxmax()
    return top_action


def s2(df_original):  # return statistics 2 (Top location)
    top_location = df_original['location'].value_counts().idxmax()
    return top_location


def s3(df_original):  # return statistics 3 (Top camera)
    top_camera = df_original['camera_id'].value_counts().idxmax()
    return top_camera


def s4(df_original):  # return statistics 4 (camera that captured least actions)
    min_camera = df_original['camera_id'].value_counts().idxmin()
    return min_camera


####### The below functions will be called in app.py


def plot_all(results, action='Eating'):
    filtered_data, new_df, list_counted_data, x, y, z = helper(results)
    return [g1(results, action), g2(new_df), g3(list_counted_data), g4(x, y, z)]


def statistics_all(results):
    # Handling cursor problem for filtered data
    coll = False
    if type(results) == pymongo.cursor.Cursor:
        print("ENTERED HEREEEEEEEE\n\n\n")
        client = pymongo.MongoClient()
        db = client["data"]
        coll = db["DATA"]

        list_dict = []

        for i in results:
            list_dict.append(i.copy())

        for dictionary in list_dict:
            coll.insert_one(dictionary)

    if coll:
        results = coll

    results_data = list(results.find())
    df_original = DataFrame(results_data)
    return [s1(df_original), s2(df_original), s3(df_original), s4(df_original)]

#######################################################################################################
