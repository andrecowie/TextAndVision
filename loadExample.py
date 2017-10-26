from sklearn.externals import joblib
from textblob import TextBlob

def split_into_tokens(message):
    if type(message) is float:
        message = str(message)
    return TextBlob(message).words#  / .tags

svm = joblib.load('tweetSent.pkl')

print(svm.predict(['hi im happy']))

print(svm.predict(['hi im sad']))