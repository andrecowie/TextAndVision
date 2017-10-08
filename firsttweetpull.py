import requests, base64, json

sec = "eyWGi1pK6OuQkkN7R9Q9fqgFrlgyrJ2eNIfcrt7nWj0VnFK9wv"
key = "AazA4OW5FtDTe8u71s4PYh0WA"
bearer = key+":"+sec
auth = base64.b64encode(bearer)
headers = {"Authorization": "Basic "+auth, "Content-type":"application/x-www-form-urlencoded;charset=UTF-8"}
body = {"grant_type": "client_credentials"}
r = requests.post("https://api.twitter.com/oauth2/token", headers=headers, data=body)
access = json.loads(r.text)['access_token']
headers = {"Authorization": "Bearer "+access, "Content-type":"application/x-www-form-urlencoded;charset=UTF-8"}
        r = requests.get("https://api.twitter.com/1.1/search/tweets.json?q=bitcoin", headers=headers)
