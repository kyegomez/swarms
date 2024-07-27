import requests

url = "https://linkedin-api8.p.rapidapi.com/linkedin-to-email"

querystring = {
    "url": "https://www.linkedin.com/in/nicolas-nahas-3ba227170/"
}

headers = {
    "x-rapidapi-key": "8c6cd073d2msh9fc7d37c26ce73bp1dea6ajsn81819935da85",
    "x-rapidapi-host": "linkedin-api8.p.rapidapi.com",
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())
