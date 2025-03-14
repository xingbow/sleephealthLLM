"""
This module provides utility functions for fetching oura health data, weather data, location information, and other related data.
"""

import time
import pytz
from timezonefinder import TimezoneFinder
from datetime import datetime
import requests
from pprint import pprint
import pandas as pd
import streamlit as st

from streamlit_js_eval import get_geolocation

import globalVariable as GV

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


geolocator = Nominatim(user_agent="activity-recommender")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

WEATHER_TYPE = GV.WEATHER_TYPE
TEMP_3_CAT = GV.TEMP_3_CAT
TEMP_5_CAT = GV.TEMP_5_CAT
TIME_CAT = GV.TIME_CAT

def filter_and_sort_scores(scores, threshold):
    """
    Filter and sort scores based on a threshold.

    Args:
        scores (list): List of scores to be filtered and sorted.
        threshold (float): The threshold value for filtering scores.

    Returns:
        list: List of indices of the filtered and sorted scores.
    """
    # Filter the scores and their indices based on the threshold
    filtered_scores = [(index, score) for index, score in enumerate(scores) if score > threshold]

    # Sort the filtered scores in descending order
    sorted_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)

    # Extract the indices
    sorted_indices = [index for index, score in sorted_scores]

    return sorted_indices

def time_to_category(current_time = None):
    """
    Convert the current time to a time category.

    Args:
        current_time (datetime): The current time to be converted. If None, the current time is used.

    Returns:
        str: The time category (e.g., '6-9', '9-12').
    """
    if current_time is None:
        # Get the current time
        current_time = datetime.now()
    hour = current_time.hour

    # Determine the category based on the hour
    if 6 <= hour < 9:
        return '6-9'
    elif 9 <= hour < 12:
        return '9-12'
    elif 12 <= hour < 15:
        return '12-15'
    elif 15 <= hour < 18:
        return '15-18'
    elif 18 <= hour < 21:
        return '18-21'
    elif 21 <= hour < 24:
        return '21-24'
    else:  # This covers the overnight hours from 0 to 6
        return '0-6'

def categorize_temperature_3_cats(temp_fahrenheit):
    """
    Categorize temperature into three categories: Cold, Mild, and Hot.

    Args:
        temp_fahrenheit (float): The temperature in Fahrenheit.

    Returns:
        str: The temperature category (e.g., 'Cold', 'Mild', 'Hot').
    """
    if temp_fahrenheit < 50:
        return "Cold"
    elif 50 <= temp_fahrenheit <= 75:
        return "Mild"
    else:
        return "Hot"
    
def categorize_temperature_5_cats(temp_fahrenheit):
    """
    Categorize temperature into five categories: Cold, Cool, Mild, Warm, and Hot.

    Args:
        temp_fahrenheit (float): The temperature in Fahrenheit.

    Returns:
        str: The temperature category (e.g., 'Cold', 'Cool', 'Mild', 'Warm', 'Hot').
    """
    if temp_fahrenheit < 32:
        return "Cold"
    elif 32 <= temp_fahrenheit < 50:
        return "Cool"
    elif 50 <= temp_fahrenheit < 68:
        return "Mild"
    elif 68 <= temp_fahrenheit < 86:
        return "Warm"
    else:
        return "Hot"

def categorize_weather_from_forecast(short_forecast):
    """
    Categorize weather from a short forecast.

    Args:
        short_forecast (str): The short forecast string.

    Returns:
        str: The weather category (e.g., 'sunny', 'rain', 'clear', 'windy', 'snow').
    """
    # Convert short_forecast to lower case to ensure case-insensitive matching
    forecast_lower = short_forecast.lower()
    
    # Mapping of keywords to weather types, with "others" as a fallback
    forecast_mapping = {
        "sunny": ["sunny", "sun"],
        "rain": ["rain", "showers", "drizzle", "thunder"],
        "clear": ["clear"],
        "windy": ["windy", "breezy"],
        "snow": ["snow", "sleet", "flurries"],
    }
    
    # Iterate over the mapping to find a match
    for weather_type, keywords in forecast_mapping.items():
        if any(keyword in forecast_lower for keyword in keywords):
            return weather_type
    
    # Default to "others" if no specific match is found
    # return "others"
    return "clear"


def get_weather_nowcast(lat, lon, api_key):
    """Get current weather information from the weather API."""
    try:
        point_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}&aqi=no"
        point_response = requests.get(point_url)
        point_response.raise_for_status()
        nowcast_url = point_response.json()["current"]

        temp_cat = categorize_temperature_3_cats(nowcast_url["temp_f"])
        weather_cat = categorize_weather_from_forecast(nowcast_url["condition"]["text"])

        current_time = point_response.json()["location"]["localtime"]
        timezone = point_response.json()["location"]["tz_id"]

        return {
            "temperatureCategory": temp_cat,
            "weatherCategory": weather_cat,
            "temperature": nowcast_url['temp_f'],
            "temperatureUnit": "F",
            "shortForecast": nowcast_url['condition']['text'],
            "windspeed": nowcast_url['wind_mph'],
            "windDirection": nowcast_url['wind_dir'],
            "timestamp": current_time,
            "timeCategory": time_to_category(datetime.strptime(current_time, "%Y-%m-%d %H:%M")),
            "timezone": timezone
        }
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def get_weather_and_location_by_user_input(q, api_key):
    """Get weather and location information by user input."""
    try:
        point_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={q}&aqi=no"
        point_response = requests.get(point_url)
        point_response.raise_for_status()
        nowcast_url = point_response.json()["current"]
        location = point_response.json()["location"]

        temp_cat = categorize_temperature_3_cats(nowcast_url["temp_f"])
        weather_cat = categorize_weather_from_forecast(nowcast_url["condition"]["text"])

        current_time = point_response.json()["location"]["localtime"]
        timezone = point_response.json()["location"]["tz_id"]

        return {
            "temperatureCategory": temp_cat,
            "weatherCategory": weather_cat,
            "temperature": nowcast_url['temp_f'],
            "temperatureUnit": "F",
            "shortForecast": nowcast_url['condition']['text'],
            "windspeed": nowcast_url['wind_mph'],
            "windDirection": nowcast_url['wind_dir'],
            "timestamp": current_time,
            "timeCategory": time_to_category(datetime.strptime(current_time, "%Y-%m-%d %H:%M")),
            "timezone": timezone
        }, {
            "latitude": location["lat"],
            "longitude": location["lon"],
            "location": location["name"]
        }
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None, None

#########################
# Location utils
#########################

def get_location():
    """Get location information by browser geolocation."""
    loc  = get_geolocation()
    if loc is not None:
        lat = loc['coords']['latitude']
        lon = loc['coords']['longitude']
        location = reverse(f"{lat}, {lon}")
        return {
            "latitude": lat,
            "longitude": lon,
            "location": location.address,
        }
    else:
        return {
            "latitude": None,
            "longitude": None,
            "location": None,
        }


#########################
# Health data utils
#########################

def get_oura_data(oura_token, d_type="workout", start_date="2024-01-01", end_date="2024-05-01"):
    """
    Retrieve data from Oura Ring API.

    Args:
        oura_token (str): The Oura token.
        d_type (str): The type of data to retrieve.
        start_date (str): The start date of the data.
        end_date (str): The end date of the data.

    Returns:
        pd.DataFrame: The data.
    """
    url = f'https://api.ouraring.com/v2/usercollection/{d_type}' 
    
    params={ 
        'start_date': start_date, 
        'end_date': end_date
    }
    headers = { 
    'Authorization': f'Bearer {oura_token}' 
    }
    response = requests.request('GET', url, headers=headers, params=params)
    if d_type != "personal_info":
        return pd.DataFrame(response.json()["data"])
    else:
        return response.json()


def get_health_data(oura_token, start_date="2024-01-01", end_date="2024-05-01"):
    """Retrieve and process comprehensive health data from Oura Ring.
    
    Args:
        oura_token (str): The Oura token.
        start_date (str): The start date of the health data.
        end_date (str): The end date of the health data.

    Returns:
        pd.DataFrame: The health data.
    """
    if oura_token is None:
        return None
    # get health data
    try:
        sleep = get_oura_data(oura_token, "sleep", start_date, end_date)[["day", "time_in_bed", "bedtime_start", "bedtime_end", "total_sleep_duration", "efficiency", "average_breath", "average_hrv", "lowest_heart_rate", "type"]]
    except:
        return None
    # Aggregate by day and sum total_sleep_duration
    # aggregated_sleep = sleep.groupby('day')['total_sleep_duration'].sum().reset_index()
    sleep.rename(columns={
        "total_sleep_duration": "total_sleep_duration (seconds)",
        "time_in_bed": "time_in_bed (seconds)",
        "efficiency": "sleep_efficiency",
        "type": "sleep_type (long_sleep: 3+ hours, sleep: naps <3 hours)"
        }, inplace=True)
    # sleep = pd.merge(sleep, aggregated_sleep, on='day', how='left')
    # sleep = sleep.drop(columns=['total_sleep_duration'])
    sleep.rename(columns={
        "average_breath": "average_breath (during sleep)", 
        "average_hrv": "average_hrv (during sleep)", 
        "lowest_heart_rate": "lowest_heart_rate (during sleep)"
        }, inplace=True)
    try:
        activity = get_oura_data(oura_token, "daily_activity", start_date, end_date)[["day", "score"]]
    except:
        activity = pd.DataFrame(columns=["day", "score"])
    activity = activity.rename(columns={"score": "activity_score (1-100)"})
    try:
        d_readiness = get_oura_data(oura_token, "daily_readiness", start_date, end_date)[["day", "score"]]
    except:
        d_readiness = pd.DataFrame(columns=["day", "score"])
    d_readiness = d_readiness.rename(columns={"score": "readiness_score (1-100)"})
    try:
        d_stress = get_oura_data(oura_token, "daily_stress", start_date, end_date)[["day", "day_summary"]]
    except:
        d_stress = pd.DataFrame(columns=["day", "day_summary"])
    d_stress = d_stress.rename(columns={"day_summary": "stress_level"})
    try:
        workout = get_oura_data(oura_token, "workout", start_date, end_date)[["day", "activity"]]
        workout = workout.groupby('day')['activity'].agg(list).reset_index()
    except:
        workout = pd.DataFrame(columns=["day", "activity"])
    # merge health data
    health_data = sleep \
    .merge(activity, "left", on="day") \
    .merge(d_readiness, "left", on="day") \
    .merge(d_stress, "left", on="day") \
    .merge(workout, "left", on="day")
    return health_data



if __name__ == "__main__":
    # get current location
    # loc_info = get_location()
    # get weather forecast
    latitude = 40.7
    longitude = -74.0
    print("location:", reverse(f"{latitude}, {longitude}"))
    forecast = get_weather_nowcast(latitude, longitude, "input_your_weatherapi_key")
    print(f"Current weather: {forecast}")