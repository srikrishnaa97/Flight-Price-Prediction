from flask import Flask, render_template, request
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

app = Flask(__name__)

with open('flight_price_regression','rb') as f:
    model = pickle.load(f)

with open('feature_scaler','rb') as f:
    scaler = pickle.load(f)

with open('one_hot_encoder','rb') as f:
    encoder = pickle.load(f)

features = []
with open('features.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        features.append(line.strip())

cities = []
with open('cities.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        cities.append(line.strip())

flight_carriers = []
with open('flight_carriers.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        flight_carriers.append(line.strip())

cat_feats = []
with open('categorical.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        cat_feats.append(line.strip())

ord_feats = []
with open('ordinal.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        ord_feats.append(line.strip())

num_feats = []
with open('numerical.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        num_feats.append(line.strip())

def none_empty(input):
    if input.strip() == '':
        return '<None>'
    else:
        return input.strip()

@app.route('/')
def index():
    return render_template('index.html',prediction=[],features=features,cities=cities,flight_carriers=flight_carriers)

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        prediction = []
        try:
            source = none_empty(request.form['Source'])
            destination = none_empty(request.form['Destination'])

            l1 = none_empty(request.form['Layover1'])
            l2 = none_empty(request.form['Layover2'])
            l3 = none_empty(request.form['Layover3'])

            crawl_date = request.form['Crawl Date']
            dep_date = request.form['Departure Date']
            dep_time = request.form['Departure Time']
            arr_date = request.form['Arrival Date']
            arr_time = request.form['Arrival Time']
            
            carr1 = none_empty(request.form['Flight_Carrier1'])
            carr2 = none_empty(request.form['Flight_Carrier1'])
            carr3 = none_empty(request.form['Flight_Carrier1'])
            carr4 = none_empty(request.form['Flight_Carrier1'])

            s_d = '<None>'
            s_l1 = '<None>'
            l1_d = '<None>'
            l1_l2 = '<None>'
            l2_d = '<None>'
            l2_l3 = '<None>'
            l3_d = '<None>'
            num_stops = 0
            
            dep_datetime = datetime.strptime(dep_date + " " + dep_time,"%Y-%m-%d %H:%M")
            arr_datetime = datetime.strptime(arr_date + " " + arr_time,"%Y-%m-%d %H:%M")
            crawl_date = datetime.strptime(crawl_date, "%Y-%m-%d")

            fli_time = divmod((arr_datetime - dep_datetime).seconds,3600)
            fli_hours = fli_time[0]
            fli_mins = fli_time[1]
            time_taken = fli_hours*60 + fli_mins

            
            crawl_dep_days = (dep_datetime - crawl_date).days

            if source == '<None>' or destination == '<None>' or source == destination:
                raise Exception("Source and Destination are the same!")
            
            if source == l1 or l3 == destination or l1 == destination or l2 == destination:
                raise Exception("Invalid layovers/destination!")
            
            if dep_datetime >= arr_datetime:
                raise Exception("Departure must be before arrival")
            
            if crawl_date >= dep_datetime:
                raise Exception("Date of search must be before departure")

            if l1 == '<None>':
                l2 = '<None>'
                carr2 = '<None>'
                if carr1 == '<None>':
                    raise Exception("Select Flight Carrier 1-1!")
                
                s_d = carr1
                num_stops = 0
            
            if l2 == '<None>':
                l3 = '<None>'
                carr3 = '<None>'
                if l1 != '<None>' and carr2 == '<None>':
                    raise Exception("Select Flight Carrier 2-1!")
                if l1 != '<None>' and carr1 == '<None>':
                    raise Exception("Select Flight Carrier 1-2!")
                s_l1 = carr1
                l1_d = carr2
                num_stops = 1
            
            if l3 == '<None>':
                carr4 = '<None>'
                if l2 != '<None>' and carr3 == '<None>':
                    raise Exception("Select Flight Carrier 3-1!")    
                if l2 != '<None>' and carr2 == '<None>':
                    raise Exception("Select Flight Carrier 2-2!")
                if l2 != '<None>' and carr1 == '<None>':
                    raise Exception("Select Flight Carrier 1-3!")
                
                s_l1 = carr1
                l1_l2 = carr2
                l2_d = carr3
                num_stops = 2

            if l3 != '<None>' and carr4 == '<None>':
                raise Exception("Select Flight Carrier 4-1!")
            else:
                s_l1 = carr1
                l1_l2 = carr2
                l2_l3 = carr3
                l3_d = carr4
                num_stops = 3
            
            dep_datetime = pd.to_datetime(pd.Series(dep_datetime))
            arr_datetime = pd.to_datetime(pd.Series(arr_datetime))

            dep_month = int(dep_datetime.dt.month[0])
            dep_week = int(dep_datetime.dt.isocalendar().week[0])
            dep_hour = int(dep_datetime.dt.hour[0])
            dep_min = int(dep_datetime.dt.minute[0])
            dep_dow = int(dep_datetime.dt.dayofweek[0])

            arr_month = int(arr_datetime.dt.month[0])
            arr_week = int(arr_datetime.dt.isocalendar().week[0])
            arr_hour = int(arr_datetime.dt.hour[0])
            arr_min = int(arr_datetime.dt.minute[0])
            arr_dow = int(arr_datetime.dt.dayofweek[0])

            
            cat_vars = [source,l1,l2,l3,destination,s_d,s_l1,l1_d,l1_l2,l2_d,l2_l3,l3_d]
            ord_vars = [num_stops,dep_month,dep_week,dep_hour,dep_min,dep_dow,arr_month,arr_week,arr_hour,arr_min,arr_dow,crawl_dep_days]
            num_vars = [time_taken]
            ordnum_vars = ord_vars+num_vars

            cat_vars = encoder.transform(np.array(cat_vars).reshape(1,len(cat_vars)))
            ordnum_vars = scaler.transform(np.array(ord_vars+num_vars).reshape(1,len(ordnum_vars)))

            all_features = np.concatenate((cat_vars,ordnum_vars),axis=1)
            prediction = int(model.predict(all_features.reshape(1,-1))[0])
            #render_template('index.html',prediction=prediction,features=features,cities=cities,flight_carriers=flight_carriers)
        except Exception as e:
            prediction = e
        finally:
            return render_template('index.html',prediction=prediction,features=features,cities=cities,flight_carriers=flight_carriers)
    else:
        return render_template('index.html',features=features,cities=cities,flight_carriers=flight_carriers)


if __name__ == '__main__':
    app.run(debug=True)