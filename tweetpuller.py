import twitter, boto3, json, requests
from creds import consumerkey, consumersecret, accesstoken, accesssecret
client = boto3.client('dynamodb')
api = twitter.Api(consumer_key=consumerkey, consumer_secret=consumersecret, access_token_key=accesstoken, access_token_secret=accesssecret)
r = requests.get("https://api.coinmarketcap.com/v1/ticker/?limit=400")
r = json.loads(r.text)
coins = []
for x in r:
    coins.append(str("$"+x["symbol"]))
stream = api.GetStreamFilter(track=coins, filter_level='low')
counter = 0
batchobject =[]
for z in stream:
    batchobject.append({'PutRequest':{'Item':{'id':{'S':str(z['id'])},'tweet':{'S':str(z["text"].encode('utf-8'))},'username':{'S':str(z["user"]["name"].encode('utf-8'))}, 'userscreenname':{'S':str(z["user"]["screen_name"].encode('utf-8'))}, 'userinfluence':{'N':str(int(z['user']['followers_count'])+int(z['user']['friends_count']))}, 'userlocation':{'S':str(z['user']['location'].encode('utf-8')) if z['user']['location'] else "None"}}}})
    if len(batchobject) == 25:
        client.batch_write_item(RequestItems={'tweets':batchobject})
        print("Wrote 25 tweets")
        batchobject = []
