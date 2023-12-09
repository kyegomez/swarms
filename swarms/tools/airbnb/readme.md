# Airbnb Service

Contributor: [Kunlun Zhu](https://github.com/Kunlun-Zhu)

You can get your RAIPID key here: https://rapidapi.com/hub

You ought to subscribe 'Airbnb API' in your account to use this tool

# Short-term Rental and Housing Information Tool

This tool, named `Short-term Rental and Housing Information`, is designed to interact with the Airbnb API to search for properties, get property details, check availability, get property reviews, and retrieve the checkout price. The tool operates by making HTTP requests to the Airbnb API and formatting the responses into an easily usable form.

## Main Functionality

1. **Search for Properties**: This functionality allows you to search for properties based on a variety of parameters like the number of adults, children, and infants, property type, amenities, check-in and check-out dates, and many more. This is done using the `search_property` function. 

2. **Search Property by Coordinates**: This function allows you to search for properties in a specific geographic area defined by the northeast and southwest coordinates of the area. This is done using the `search_property_by_coordinates` function.

3. **Search for Destination**: The `search_destination` function helps to perform a destination search given a query and optionally a country. It returns positions 'ID' information.

4. **Get Property Details**: The `get_property_details` function is used to retrieve detailed information about a specific property. This includes the number of rooms, amenities, location, and other relevant information.

5. **Check Property Availability**: This function, `check_availability`, allows you to check if a property is available for booking. 

6. **Get Property Reviews**: You can use the `get_property_reviews` function to retrieve reviews of a property. 

7. **Get Property Checkout Price**: The `get_property_checkout_price` function is used to get the checkout cost of a property given its ID and check-in date.

This tool provides a simple and effective way to interact with the Airbnb API, making it easier for developers to incorporate Airbnb data into their applications.