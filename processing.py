import json
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from textblob import TextBlob
import pickle


def main():
	model = pickle.load(open('tonDreSent.pkl', 'rb'))
	with open('namedcoins.json') as f:
		coins = json.loads(f.read())
	with open('newdata.json') as f:
		data = json.loads(f.read())
	allNes = " "
	coincounta = {}
	for x in data:
		tweet = x[1]['attributes']['tweet']['S']
		oursentiment = model.predict([tweet])
		a = tokneiza(tweet)
		words = list(a[0])
		for word in words:
			for coin, coinphrase in coins.items():
				if word in coinphrase:
					if coin in list(coincounta.keys()):
						coincounta[coin][0] += 1
						coincounta[coin][1].append(oursentiment[0])
					else:
						coincounta[coin]= [1, [oursentiment[0]]]
		nouns = list(a[2])
		allNes += " ".join(nouns)
	allNes.maketrans(" ","$")
	allNes.maketrans(" ","#")
	allNes.maketrans(" ", "@")
	fd = FreqDist(word_tokenize(allNes))
	print(fd.most_common(100))
	for coin, mentions in coincounta.items():
		print(coin+" was mentioned: "+str(mentions[0])+" times, "+str(sum(mentions[1]))+" times positively")


def split_into_tokens(message):
    if type(message) is float:
        message = str(message)
    return TextBlob(message).words#  / .tags

def tokneiza(message):
	if type(message) is not str:
		message = str(message)
	mess = TextBlob(message)
	return mess.words, mess.tags, mess.noun_phrases, mess.sentiment

	# tweetCorpus = ", ".join([x[1]['attributes']['tweet']['S'] for x in data])
	# fd = FreqDist(word_tokenize(tweetCorpus))
	# fd2 = FreqDist(tweetCorpus.split(' '))
	# print(fd.most_common(100))
	# print(fd2.most_common(100))



if (__name__ == '__main__'):
	main()
