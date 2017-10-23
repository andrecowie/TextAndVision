import json
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def main():
	with open('data.json') as f:
		data = json.load(f)
	tweetCorpus = ", ".join([x['tweet'] for x in data])
	fd = FreqDist(word_tokenize(tweetCorpus))
	fd2 = FreqDist(tweetCorpus.split(' '))
	print(fd.most_common(100))
	print(fd2pp.most_common(100))



if (__name__ == '__main__'):
	main()
