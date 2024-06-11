GLOSSARY_PROMPTS = """

Glossary

API Terminology
Access_key
A private access key or shared access key (not a secret and not an Application Key) used to access the Baron Weather API. View your access keys on your account page.

Application Key
Users’ personal and confidential key from which access keys are derived. The application key allows management of access keys. View your application key on your account page.

Configuration_code
Configuration codes are most often used to differentiate between EPSG:3857 (Mercator) and EPSG:4326 (Geodetic) projections. In the Baron Weather API we add a descriptor to the beginning to indicate any additional parameters to the projection. The default descriptor is ‘Standard’ and will be the primary configuration used, but some data products may offer alternative descriptor to differentiate formatting options.

Coordinated Universal Time (UTC)
This standard organizes the data so the largest temporal term (the year) appears first in the data string and progresses to the smallest term (the second), like so 2012-12-31TI8:51:23Z.

Format
The language format for API responses. In the Baron Weather API, responses for text products can be in JSON or JSONP format, and graphical formats are always in png format.

ISO8601
The primary time standard by which the world regulates clocks and time.

Max-age
It's an optional parameter for the metar, buoy, and cwop "nearest" api which allows developers to query a lat/lon and only get back data is more recent than the prescribed date and time.

Metadata_timestamp
The ISO 8601 UTC date/time for the data found in the returned metadata "time" parameter(s).

Metadata_valid_time
The ISO 8601 UTC date/time for the data found in the returned metadata "valid_times" list. This is required for forecast products (those that provide a valid_times list in the metadata), but unnecessary for non-forecast products.

Pages
The page parameter was put in place to minimize the amount of information returned in the response. Text products that support the page parameter return the current page number and the total number of pages when you make a request. Many text products provide thousands of lines of data, which can be overwhelming when users are looking for a specific piece of information for a specific time frame. For example, a developers looking for the current weather conditions at all METAR stations will not need to have thousands of lines of text returned. Instead, we limit them to a maximum number of stations per page, then if users want the full set, they have to ask explicitly for page 2, page 3, etc. in the request URL.

Product Code
The code to include in the API URL request that is specific to each weather product.

Reference Time
The time the forecast model begins. In the product-instances metadata, this is called "time".

Timestamp
The timestamp value included with the request and used to create the signature. Represented as ‘ts’ in request and always in UTC format.

Timestep
In general, a single point in time for which the product is valid, also called "valid_times". However for accumulation products, the timesteps represent the end of a measured time interval for which total accumulated precipitation is forecast. A list of timesteps or "valid_times" are provided In the product-instances metadata.

Timestep Interval
The interval between timesteps.

Valid_times
The list of UTC-formatted timesteps for a forecast product when the Product Instances API is run.

X
The x-coordinate of the requested tile. This value represents the horizontal index of the tile, assuming an origin of the lower left corner of the tile grid (0,0). These coordinates correspond to the Tile Map Service Specification.

Y
The y-coordinate of the requested tile. This value represents the vertical index of the tile, assuming an origin of the lower left corner of the tile grid (0,0). These coordinates correspond to the Tile Map Service Specification.

Z
The z-coordinate of the requested tile. This value represents the zoom level (depth) of the tile. A value of 0 shows the entire world using the minimum number amount of tiles (1 for Mercator, 2 for Geodetic). The maximum available depth may vary by product. These coordinates correspond to the Tile Map Service Specification.

 
 

Meteorological Terminology
dBZ
Stands for decibels relative to Z. It is a meteorological measure of equivalent reflectivity (Z) of a radar signal reflected off a remote object.

Dew Point
The temperature below which the water vapor in a volume of humid air at a constant barometric pressure will condense into liquid water.

Heat Index
An index that combines air temperature and relative humidity in an attempt to determine the human-perceived equivalent temperature — how hot it feels.

Infrared (IR)
In relation to satellite imagery, infrared imagery is produced by satellite analysis of infrared wavelengths. This analysis indicates the temperature of air masses, making it possible to identify cloud cover day or night.

kft
Stands for thousands of feet.

Relative Humidity
The ratio of the partial pressure of water vapor in an air-water mixture to the saturated vapor pressure of water at a given temperature.

Valid Time Event Code (VTEC)
Format in which alerting information is pulled from the National Weather Service.

Visible Satellite (VIS)
Visible satellite imagery is a snapshot of cloud cover from space. Consequently it is only usable during daylights hours. It is the easiest weather data product for laypeople to understand.

Warnings
The NWS issues a warning when a hazardous weather or hydrologic event is occurring, is imminent, or has a very high probability of occurring. Often warnings are not issued until conditions have been visually verified. A warning is used for conditions posing a threat to life or property.

Watches
The NWS issues a watch when the risk of a hazardous weather or hydrologic event has increased significantly, but its occurrence, location, and/or timing is still uncertain. It is intended to provide enough lead time so that those who need to set their plans in motion can do so.

Water Vapor Satellite
Water vapor imagery is a satellite product which measures the amount of moisture in the atmosphere above 10,000 feet. Bright white areas indicate abundant moisture, which may be converted into clouds or precipitation. Darker areas indicate the presence of drier air. In addition to measuring moisture, water vapor imagery is useful in detecting large scale weather patterns, such as jet streams.

Wave Dominant Period
The period in seconds between successive waves.

Wave Height
The maximum reported or forecasted wave height.

Wind Chill
The perceived decrease in air temperature felt by the body on exposed skin due to the flow of cold air. Wind chill temperature is defined only for temperatures at or below 10 °C (50 °F) and wind speeds above 4.8 kilometers per hour (3.0 mph).

Wind Gust
A sudden, brief increase in speed of wind. According to US weather observing practice, gusts are reported when the peak wind speed reaches at least 16 knots and the variation in wind speed between the peaks and lulls is at least 9 knots. The duration of a gust is usually less than 20 seconds.

"""

WEATHER_AGENT_SYSTEM_PROMPT = """

You navigate through tasks efficiently. Whether you're learning something new or need assistance with daily tasks, I can provide information, suggestions, and step-by-step guidance.

#### How I Can Help:
- **Information Retrieval:** I can fetch and summarize information on a wide range of topics.
- **Problem Solving:** I offer solutions and strategies to address specific challenges.
- **Learning Support:** I assist in understanding new concepts and procedures.

#### Example: Using the Baron Weather API

Let's look at how you can use the Baron Weather API to retrieve weather data, which involves making authenticated HTTP requests.

1. **Understand Your Needs**: Identify what specific weather data you need, such as current conditions or a forecast.
2. **Gather API Details**: Know your API key, the endpoints available, and the data format (JSON).
3. **Authentication**: Learn how to authenticate your requests using your API key and additional security measures as required (like generating signatures).
4. **Craft the Request**: Construct the correct HTTP request to fetch the data you need.
5. **Parse the Response**: After making the request, interpret the JSON response to extract and utilize the weather data.

Through each step, I can provide explanations, code snippets, and troubleshooting tips to ensure you successfully achieve your goal.

### Conclusion

With these steps, you'll be better prepared to use tools like APIs effectively and get the most out of our interactions. If you have questions or need further assistance, feel free to ask!

---

"""


FEW_SHORT_PROMPTS = """
What is the current temperature?	allow the user to request the current temperature for their location	user's location	request_metar_nearest("38", "-96")
Describe the current weather.	have the LLM construct a narrative weather description based on current conditions	user's location	request_metar_nearest("38", "-96")
How much rain fell at my location?	allow the user to determine how much rain has accumulated at their location in the last 24 hours	user's location	point_query('precip-totalaccum-24hr', 'Standard-Mercator', -86.6, 34.4)
Is it going to be sunny tomorrow?	allow the user to determine cloud coverage for their location 	user's location	request_ndfd_basic(34.730301, -86.586098, forecast_time)
Is rain expected at my location in the next 6 hours?	allow the user to determine if precip will fall in the coming hours	user's location	point_query('baron-hires-maxreflectivity-dbz-all', 'Mask1-Mercator', -86.6, 34.4)
What is the max forecasted temperature today? 	allow the user to determine how hot or cold the air temp will be	user's location	request_ndfd_basic(34.730301, -86.586098, forecast_time)
Will it be windy today? 	allow the user to determine the max wind speed for that day	user's location	point_query('baron-hires-windspeed-mph-10meter', 'Standard-Mercator', -86.6, 34.4)
"""
