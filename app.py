from flask import Flask, render_template, url_for, request
import joblib
from mymodel import MyModel
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Impor model yang sudah dilatih
model_x = joblib.load('modelX.joblib')

# Input pre-processing
def get_season(string_date):
    date = string_date.split('-')
    month = date[1]
    day = date[2]
    season = {'Autumn':0, 'Winter':1, 'Spring':2, 'Summer':3}
    if (month == 3 and day >= 20) or (month == 4) or (month == 5) or (month == 6 and day < 21):
        return season['Spring']
    elif (month == 6 and day >= 21) or (month == 7) or (month == 8) or (month == 9 and day < 23):
        return season['Summer']
    elif (month == 9 and day >= 23) or (month == 10) or (month == 11) or (month == 12 and day < 22):
        return season['Autumn']
    else:
        return season['Winter']

def is_holiday(string_date):
    date = string_date.split('-')
    month = date[1]
    day = date[2]

    url = 'https://www.timeanddate.com/holidays/uk/2018'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    holiday = soup.find('table', {'id':'holidays-table'})
    holiday_date = holiday.find_all('th', class_='nw')
    holiday_date = [date.text.split(' ') for date in holiday_date]

    try:
        switcher = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
        }
        formatted_date = [[switcher[date[0]], int(date[1])] for date in holiday_date]
        return (1 if [month, day] in formatted_date else 0)
    except KeyError:
        switcher = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'Mei': 5,
        'Jun': 6,
        'Jul': 7,
        'Agu': 8,
        'Sep': 9,
        'Okt': 10,
        'Nov': 11,
        'Des': 12
        }
        formatted_date = [[int(date[0]), switcher[date[1]]] for date in holiday_date]
        return (1 if [day, month] in formatted_date else 0)
    
def scale_feature(df):
    scaler = joblib.load('scaler_model.joblib')
    df[df.columns[:9]] = pd.DataFrame(scaler.transform(df[df.columns[:9]]),columns = df.columns[:9])


# App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    date = request.form['date']
    hour = request.form['hour']
    temp = request.form['temp']
    humidity = request.form['humidity']
    wind_speed = request.form['wind_speed']
    visibility = request.form['visibility']
    dew_temp = request.form['dew_temp']
    solar_rad = request.form['solar_rad']
    rainfall = request.form['rainfall']
    snowfall = request.form['snowfall']
    func_no = 1 if request.form['func'] == 'No' else 0
    func_yes = 1 if request.form['func'] == 'Yes' else 0 
    season = get_season(date)
    holiday_no = int(not is_holiday(date))
    holiday_yes = int(is_holiday(date))
    df_feature = pd.DataFrame([{'Hour':hour, 
                                'Temperature(°C)':temp, 
                                'Humidity(%)':humidity, 
                                'Wind speed (m/s)':wind_speed, 
                                'Visibility (10m)':visibility, 
                                'Dew point temperature(°C)':dew_temp, 
                                'Solar Radiation (MJ/m2)':solar_rad, 
                                'Rainfall(mm)':rainfall, 
                                'Snowfall (cm)':snowfall, 
                                'Season':season, 
                                'FunctioningDay_No':func_no, 
                                'FunctioningDay_Yes':func_yes, 
                                'Holiday_No':holiday_no, 
                                'Holiday_Yes':holiday_yes}])
    #print(df_feature.head())
    scale_feature(df_feature)
    #print(df_feature.head())
    pred = model_x.predict(df_feature)
    return render_template("prediction.html", date=date,
                                              hour=hour,
                                              temp=temp,
                                              humidity=humidity,
                                              wind_speed=wind_speed,
                                              visibility=visibility,
                                              dew_temp=dew_temp,
                                              solar_rad=solar_rad,
                                              rainfall=rainfall,
                                              snowfall=snowfall,
                                              func_no=func_no,
                                              func_yes=func_yes,
                                              season=season,
                                              holiday_no=holiday_no,
                                              holiday_yes=holiday_yes,
                                              pred=round(pred[0]))

if __name__ == "__main__":
    app.run()