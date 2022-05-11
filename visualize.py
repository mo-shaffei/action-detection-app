import plotly
import plotly.express as px
from pandas import DataFrame
import plotly.graph_objects as go
import json
from bson.json_util import dumps


def plots(results, action='eating'):

    def g1(action):
        
        filtered_data = results.aggregate([
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

        filtered_data = results.aggregate([
            {
                "$group" : {
                    "_id" :  {  "location": "$location", "action": "$action"} ,
                    "count": { "$sum" : 1 } }
            },

            {
                "$group" : {"_id" : "$_id.location", "actions": { 
                "$push": { "action":"$_id.action", "count":"$count" }
            } } 
            }
        ])
        
        list_data = list(filtered_data)
        df = DataFrame(list_data)
        x=[]; y=[]; z=[]; i=-1
        for location in df._id:
            i=i+1
            for item in df.actions[i]:
                x.append(item.pop('action'))
                y.append(item.pop('count'))
                z.append(location)

        new_df=DataFrame(list(zip(x, y, z)),
                columns=['action','count', 'location'])

        fig2 = px.bar(new_df, x="location", y="count", color='action',
        barmode='group')

        graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        return graph2JSON


    def g3():
        counted_actions = results.aggregate([
            {
                "$group" : {"_id" : "$action", "count": {"$sum" : 1}}  
            }
        ])

        list_data = list(counted_actions)
        df = DataFrame(list_data)

        fig3 = px.pie(df, values='count', names='_id')

        graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
        return graph3JSON

    def g4():
        filtered_data = results.aggregate([
                {
                    "$group" : {
                        "_id" :  {  "location": "$location", "action": "$action"} ,
                        "count": { "$sum" : 1 } }
                },

                {
                    "$group" : {"_id" : "$_id.location", "actions": { 
                    "$push": { "action":"$_id.action", "count":"$count" }
                } } 
                }
            ])
        
        list_data = list(filtered_data)    
        df = DataFrame(list_data)
        x=[]; y=[]; z=[]; i=-1
        for location in df._id:
            i=i+1
            for item in df.actions[i]:
                x.append(item.pop('action'))
                y.append(item.pop('count'))
                z.append(location)

        fig4 = go.Figure(data=go.Heatmap( {'z':y, 'x':x, 'y':z} ))

        graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
        return graph4JSON

    return [g1(action), g2() , g3(), g4()]