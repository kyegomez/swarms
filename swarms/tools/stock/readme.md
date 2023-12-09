# Stock Market Service

Contributor: [Kunlun Zhu](https://github.com/Kunlun-Zhu)

You can get your free Alpha Vantage key here: https://www.alphavantage.co/support/#api-key

# Stock Info Tool

This tool allows you to look up stock information.

## Setup

The tool is initialized with the following parameters:

- **name_for_model**: "Stock Info"
- **description_for_model**: "Plugin for look up stock information"
- **logo_url**: "https://your-app-url.com/.well-known/logo.png"
- **contact_email**: "hello@contact.com"
- **legal_info_url**: "hello@legal.com"

## Endpoint

The tool provides the following endpoints:

- **/get_today_date**: Get today's date.
- **/add_date**: Add days to a date. Date should be pass as 'yyyy-mm-dd'.
- **/get_daily_prices**: Get the stock price of an entity in the stock market. Date should be pass as 'yyyy-mm-dd'.
- **/get_open_info**: Get information about if the market in the region is open.
- **/get_exchange_rate**: This API returns the realtime exchange rate for a pair of digital currency (e.g., Bitcoin) and physical currency (e.g., USD).

## Function Descriptions

- **get_today_date()**: This function returns today's date.
- **add_date(date: str, days: int)**: This function adds a certain number of days to a date. The date should be passed as 'yyyy-mm-dd'.
- **get_daily_prices(symbol: str, date: str = '')**: This function gets the stock price of an entity in the stock market. The date should be passed as 'yyyy-mm-dd'.
- **get_open_info(region: str = 'United States')**: This function gets information about if the market in the region is open.
- **get_exchange_rate(from_currency: str = 'USD', to_currency: str = 'BTC')**: This function returns the realtime exchange rate for a pair of digital currency (e.g., Bitcoin) and physical currency (e.g., USD).