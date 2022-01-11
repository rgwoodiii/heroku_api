import requests
import json


# URL that resolves to workspace
URL = "http://127.0.0.1:8000"

# call APIs
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

print(requests.get("%s/" % URL, headers=headers).text)

payload = json.dumps({
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })

response = requests.request("POST", URL, headers=headers, data=payload)
print(response.text)
