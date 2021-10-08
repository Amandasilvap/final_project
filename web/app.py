#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import pandas as pd
import getpass
# from __future__ import print_function
import json
import sys
from random import randint
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn import cluster
import sklearn.metrics as metrics
from sklearn.metrics import silhouette_score
pd.set_option('display.max_columns', None)
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestCentroid
from datetime import datetime
from flask import Flask, request


weather_country_monthly = pd.read_csv("/Users/amandamac/IronHackerDA/FINAL PROJECT/weather_country_monthly.csv")
weather_model = pd.read_csv("/Users/amandamac/IronHackerDA/FINAL PROJECT/weather_model.csv")

# Scaling
minmax_scaler = MinMaxScaler().fit(weather_model)
X = minmax_scaler.transform(weather_model)
X_prep = pd.DataFrame(X,columns=weather_model.columns).head()


# ##### Cluster

kmeans = KMeans(n_clusters=6, n_init = 50, max_iter=100, random_state = 1234)
y_kmeans = kmeans.fit_predict(X)
# Predicting / assigning the clusters:
clusters = kmeans.predict(X)
# Check the size of the clusters
pd.Series(clusters).value_counts().sort_index()

# Calculate Scores
score = silhouette_score(X, kmeans.labels_, metric='euclidean')
print(metrics.calinski_harabasz_score(X, kmeans.labels_))
print('Silhouetter Score: %.3f' % score)
print(kmeans.inertia_)
kmeans.score(X)

# ### Putting all Together
clusters = pd.Series(clusters)
weather_cluster = pd.concat([weather_country_monthly, clusters], axis=1)
weather_cluster = weather_cluster.rename(columns={0: 'cluster_kmeans'})
weather_cluster
weather_cluster.to_csv('weather_cluster.csv')

key = 'b98ec06ca2eb4805893132109212909'

def getWeather(city, month):
    try:
        response = requests.get(
            url="https://api.worldweatheronline.com/premium/v1/past-weather.ashx",
            params={"key": key, "type": "city", "q": city, "format": "json", "date": "%s-01" % month,"enddate": "%s-30" % month,
                "tp": "24",},).json()                   
        return response
    
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

        
def Weather_Usersearch(city, month):
    columns = ['moon_illumination', 'maxtempC', 'mintempC', 'avgtempC', 'totalSnow_cm', 'sunHour', 'uvIndex', 
               'windspeedKmph', 'winddirDegree','precipMM', 'humidity', 'visibility', 'pressure', 'cloudcover', 'HeatIndexC', 'DewPointC', 
               'WindChillC', 'WindGustKmph']
    weather_user = pd.DataFrame(columns= columns)

    year = 2021
    current_month = datetime.today().strftime('%m')
    if int(month) > int(current_month):
        year = int(datetime.today().strftime('%Y')) - 1
    else:
        year= int(datetime.today().strftime('%Y'))

    weather = getWeather(city=city, month="%i-%s" % (year, month))   
    weather = weather['data']['weather']

    for item in weather:
        row = {}
        for column in columns:
            if column in item:
                row[column] = item[column]
            elif column in item['hourly'][0]:
                row[column] = item['hourly'][0][column]
            elif column in item['astronomy'][0]:
                row[column] = item['astronomy'][0][column]
            else:
                print('Column [%s] not found on item [%s]' % (column, item))

        weather_user = weather_user.append(row, ignore_index=True)
        weather_user = weather_user.apply(pd.to_numeric)
        weather_user = weather_user.mean().round(0).to_frame().T
        weather_user_cluster = minmax_scaler.transform(weather_user)
        weather_user_cluster = kmeans.predict(weather_user_cluster)
        weather_user_cluster = pd.Series(weather_user_cluster).values[0]
    return weather_user_cluster


# Flights API


rapidAPIkey = '63c39fc192msh7a55c70997b5229p1c4978jsn02225d436221'

def city_code(city_name):
    url = "https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/autosuggest/v1.0/US/USD/en-US/"
    params = {"query": city_name}
    headers = {'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com",
               'x-rapidapi-key': rapidAPIkey}
    response = requests.get(url, headers = headers, params = params)
    if len(response.json()["Places"]) > 0:
        return response.json()["Places"][0]["PlaceId"]
    else:
        return -1

def flight_prices(departure, arrival, date):
    departure_code = city_code(departure)
    arrival_code = city_code(arrival)
    
    if departure_code == -1 or arrival_code == -1:
        return {}
    
    url = f"https://skyscanner-skyscanner-flight-search-v1.p.rapidapi.com/apiservices/browsequotes/v1.0/DE/EUR/en-US/{departure_code}/{arrival_code}/{date}"
    params = {"inboundpartialdate":{date}}
    headers = {
    'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com",
    'x-rapidapi-key': rapidAPIkey}
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def get_dates(start, end):
    return pd.Series(pd.date_range(start, end,freq='d').format())

def flight_days(origin, destination, start, end):
    dates = get_dates(start, end)
    return {date:flight_prices(origin, destination, date) for date in dates}

def lowest_price(origin, destination, date):
    flights = flight_prices(origin, destination, date)
    min_quote = {'MinPrice': 99999}
    if "Quotes" not in flights:
        return {}
    
    for quote in flights["Quotes"]:
        if quote['MinPrice'] < min_quote['MinPrice']:
            min_quote = quote

    if 'OutboundLeg' not in min_quote:
        return {}
    
    price = {'date': min_quote['OutboundLeg']['DepartureDate'][:10],
             'price': min_quote["MinPrice"]}
    return price
    #return min(prices, key = prices.get)


# Building the Recommender

# print("Hello! Let me give you suggestions for your next trip!","\n")
# sleep(1)

# print("Enter your favorite city, or somewhere you've already visited.","\n")
# sleep(1)

# city_user = input("City name: ")

# print("\n","Think of the month when the weather was nice in this place","\n")
# sleep(1)

# month= int(input("Month: "))

def build_recommendation(city_user, month):

    choice = weather_cluster.loc[(weather_cluster['city'].str.lower() == city_user.lower()) & (weather_cluster['month'] == month)]

    response = ''
    list_result = []

    if len(choice) > 0:
        suggest = weather_cluster.loc[weather_cluster['cluster_kmeans'] == choice['cluster_kmeans'].values[0]].sample(5)
        response = response + "<p>These cities can be a good choice:</p><br />"
        for index, item in suggest.iterrows():
            response = response + "<p>You can choose " + item['city'] + " is located " + item['country'] + " and they speak: " + item['language '] + " and the currency is " + item['currency_name'] + "<br />"
            list_result.append(item['city'])
    else:
        weather_user_cluster = Weather_Usersearch(city_user, month)
        if weather_user_cluster >=0:
            weather_cluster.loc[weather_cluster['cluster_kmeans'] == weather_user_cluster].sample(5)
            response = response + "<p>These cities can be a good choice too:</p><br />"
            for index, item in suggest.iterrows():
                response = response + "<p>You can choose " + item['city'] + " is located " + item['country'] + " and they speak: " + item['language '] + " and the currency is " + item['currency_name'] + "<br />"
                list_result.append(item['city'])
        else:
            response = response + "<p>Not Found this city</p><br />"

    return (response, list_result)

def build_flights(current_city, month, list_result):
    # current_city = input("Current city name: ","\n")

    response = ""
    for city in list_result:
        price = lowest_price(current_city, city, "2022-%02d" % month)
        if 'price' in price:
            response = response + ('<p>Lowest price for %s is %i EUR in %s </p><br />' % (city, price['price'], price['date']))
        else:
            response = response + ('<p>Flights not found for %s </p><br />' % city)

    return response


app = Flask(__name__)

html = """
<style>
.box {{
        background-color:white;
        color:black;
        
        margin:120px auto;
           
        width: 400px;
    }}
    p {{ margin: 10px 20px }}
    body {{
background-color:#2dc5fa;
    }}
    </style>
<body>
    <div class=box>
    <br />
        {code}
        <br />
    </div>
    </body>
"""

@app.route("/")
def start():
    # print("Hello! Let me give you suggestions for your next trip!","\n")
# sleep(1)

# print("Enter your favorite city, or somewhere you've already visited.","\n")
# sleep(1)

# city_user = input("City name: ")

# print("\n","Think of the month when the weather was nice in this place","\n")
# sleep(1)

# month= int(input("Month: "))
    code = """  
<p>Hello! Let me give you suggestions for your next trip!</p>
    <br />
        <form action = "/recommendation" method = "POST">
         <p>Enter your favorite city, or somewhere you've already visited
         
         <input type="text" name="city_user" /></p>
         <br /> 
         <p>Think of the month when the weather was nice in this place 
         
         <input type="text" name="month" /></p>
         <br />
         <p><input type="submit" value="Submit" /></p>
      </form>
    """

    return html.format(code=code)

@app.route("/recommendation", methods=['POST'])
def recommendation():
    result = request.form
    recommendation = build_recommendation(request.form.get('city_user'), int(request.form.get('month')))

    response = recommendation[0] + """<br /><br /><p>Now let us help to show some flights price for you for our recommendations</p>
    <br />
        <form action = "/flights" method = "POST">
         <p>Current city name
         
         <input type="text" name="current_city" /></p>
         <br />
         <input type="hidden" name="month" value="{month}" />
         <input type="hidden" name="list_result" value="{list_result}" />
         <p><input type="submit" value="Submit" /></p>
      </form>
    """.format(month=result['month'], list_result=",".join(recommendation[1]))

    return html.format(code=response)

@app.route("/flights", methods=['POST'])
def flights():
    result = request.form
    code = build_flights(request.form.get('current_city'), int(request.form.get('month')), request.form.get('list_result').split(","))
    code = code + """
    <br />
    <br />
    <p><a href="/">Try again</a></p>
    <br />
    """
    return html.format(code=code)



