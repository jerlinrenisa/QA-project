import requests

url = "http://127.0.0.1:5000/answer"
data = {
    "question": "Who developed AI?",
    "context": "Artificial intelligence (AI) is developed by OpenAI."
}

response = requests.post(url, json=data)
print(response.json())
