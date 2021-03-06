import datetime
import requests

from settings import APPID


def command_time():
    return datetime.datetime.now().time()


def command_weather() -> str:
    city_id = 1496747
    answer = requests.get(
        "http://api.openweathermap.org/data/2.5/weather",
        params={"id": city_id, "units": "metric", "lang": "ru", "APPID": APPID},
    )
    result = answer.json()
    return (
        "город: {}\n"
        "текущее состояние: {}\n"
        "температура: {}\n"
        "максимальная температура: {}\n"
        "минимальная температура: {}\n"
        "атмосферное давление: {}\n"
        "влажность: {}".format(
            result["name"],
            result["weather"][0]["description"],
            result["main"]["temp"],
            result["main"]["temp_max"],
            result["main"]["temp_min"],
            result["main"]["pressure"],
            result["main"]["humidity"],
        )
    )
