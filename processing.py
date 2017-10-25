import json
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from textblob import TextBlob

def main():
	with open('newdata.json') as f:
		x = f.read()
	data = json.loads(x)
	allNes = " "
	for x in data:
		a = tokneiza(x[1]['attributes']['tweet']['S'])
		allNes += " ".join(list(a[2]))
	allNes.maketrans(" ","$")
	allNes.maketrans(" ","#")
	allNes.maketrans(" ", "@")
	fd = FreqDist(word_tokenize(allNes))
	print(fd.most_common(100))


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
