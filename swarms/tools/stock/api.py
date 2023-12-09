import requests
import json
from datetime import date, datetime, timedelta
import os
from ..tool import Tool


def build_tool(config) -> Tool:
    tool = Tool(
        "Stock Info",
        "Look up stock information",
        name_for_model="Stock",
        description_for_model="Plugin for look up stock information",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    functions = [
        "TIME_SERIES_INTRADAY",
        "TIME_SERIES_INTRADAY_EXTENDED",
        "TIME_SERIES_DAILY",
        "TIME_SERIES_DAILY_ADJUSTED",
    ]
    types = ["open", "close", "high", "low"]

    KEY = config["subscription_key"]
    BASE_URL = "https://www.alphavantage.co/query?"

    def get_json_data(
        function,
        symbol,
        interval="5min",
        adjusted="true",
        outputsize="compact",
        datatype="json",
    ):
        url = BASE_URL + "function=" + function + "&symbol=" + symbol + "&apikey=" + KEY
        r = requests.get(url)
        data = json.loads(r.text)
        return data

    @tool.get("/get_today_date")
    def get_today_date():
        """Get today's date"""
        today = date.today()
        return today.strftime("%Y-%m-%d")

    @tool.get("/add_date")
    def add_date(date: str, days: int):
        """Add days to a date. Date should be pass as 'yyyy-mm-dd'."""
        date = datetime.strptime(date, "%Y-%m-%d")
        new_date = date + timedelta(days=days)
        return new_date.strftime("%Y-%m-%d")

    @tool.get("/get_daily_prices")
    def get_daily_prices(symbol: str, date: str = ""):
        """Get the stock price of an entity in the stock market. Date should be pass as 'yyyy-mm-dd'."""
        if "," in symbol:
            symbol, date = symbol.split(",")
        if date.strip() == "":
            return "Please specify a date and try again. You can you get_today_date to up-to-date time information."
        data = get_json_data("TIME_SERIES_DAILY_ADJUSTED", symbol)
        # print(data.keys())
        time_series = data["Time Series (Daily)"]
        final_time = ""
        print(time_series)
        # 查找最接近查询日期的数据
        for timestamp, daily_data in time_series.items():
            print(timestamp)
            if timestamp == date:
                open_price = daily_data["1. open"]
                high_price = daily_data["2. high"]
                low_price = daily_data["3. low"]
                close_price = daily_data["4. close"]
                volume = daily_data["6. volume"]
                break
            elif timestamp < date:
                final_time = timestamp
                open_price = time_series[timestamp]["1. open"]
                high_price = time_series[timestamp]["2. high"]
                low_price = time_series[timestamp]["3. low"]
                close_price = time_series[timestamp]["4. close"]
                volume = time_series[timestamp]["6. volume"]
                break
        return {
            "open": open_price,
            "close": close_price,
            "high": high_price,
            "low": low_price,
            "symbol": symbol,
            "date": final_time,
            "volume": volume,
        }

    @tool.get("/get_open_info")
    def get_open_info(region: str = "United States"):
        """get information about if the market in the region is open"""
        url = "https://www.alphavantage.co/query?function=MARKET_STATUS&apikey=" + KEY
        r = requests.get(url)
        data = json.loads(r.text)
        for item in data["markets"]:
            if item["region"] == region:
                return item["current_status"]
        return " not found"

    @tool.get("/get_exchange_rate")
    def get_exchange_rate(from_currency: str = "USD", to_currency: str = "BTC"):
        """This API returns the realtime exchange rate for a pair of digital currency (e.g., Bitcoin) and physical currency (e.g., USD)."""
        url = (
            "https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency="
            + from_currency
            + "&to_currency="
            + to_currency
            + "&apikey="
            + KEY
        )
        r = requests.get(url)
        data = json.loads(r.text)
        try:
            rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
            return rate
        except:
            return "error"

    return tool
