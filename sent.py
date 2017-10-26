import pandas
from sklearn.externals import joblib
import _pickle as cPickle
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold



def split_into_tokens(message):
    if type(message) is float:
        message = str(message)
    return TextBlob(message).words#  / .tags


def traintest(messages):
    msg_train, msg_test, label_train, label_test = \
        train_test_split(messages['SentimentText'], messages['Sentiment'], test_size=0.3)
    return [msg_train, msg_test, label_train, label_test]

#Read Kaggle / Sentiment140 data, Sanders Twitter data and STS_Gold Twitter Data
sent = pandas.read_csv('sent.csv',encoding='utf-8',error_bad_lines=False, nrows=4200)
sent2 =pandas.read_csv('senti2.csv',encoding='utf-8',error_bad_lines=False)
sent3 =pandas.read_csv('stsgold.csv',encoding='utf-8',error_bad_lines=False)


frame = [sent, sent2, sent3]
combi = pandas.concat(frame)


def tonyFunction(msg_train, msg_test, label_train, label_test, info):


    # pipeline = Pipeline([
    #     ('bow', CountVectorizer(analyzer=split_into_tokens)),  # strings to token integer counts can use #stop_words('english)
    #     ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    #     ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    # ])
    #
    # params = {
    #     'tfidf__use_idf': (True, False),
    #     }
    #
    # grid = GridSearchCV(
    #     pipeline,  # pipeline from above
    #     params,  # parameters to tune via cross validation
    #     refit=True,  # fit using all available data at the end, on the best found param combination
    #     n_jobs=1,  # number of cores to use for parallelization; -1 for "all cores"
    #     scoring='accuracy',  # what score are we optimizing?
    #     cv=StratifiedKFold(n_splits=5),  # what type of cross validation to use
    # )
    #
    # nb_detector = grid.fit(msg_train, label_train)
    # print('\nTest Results')
    # print('\nMultinomial NaiveBayes '+info)

    # predictions = nb_detector.predict(msg_test)
    # print ('\nConfusion Matrix')
    # print (confusion_matrix(label_test, predictions))
    # print (classification_report(label_test, predictions))

    #SVM Implementation
    pipeline_svm = Pipeline([
        ('bow', CountVectorizer(analyzer=split_into_tokens)),
        ('tfidf', TfidfTransformer()),
        ('classifier', SVC()),  # <== change here
    ])

    # pipeline parameters to automatically explore and tune / limited for speed now
    param_svm = [
      {'classifier__C': [1], 'classifier__kernel': ['linear']},
     ]

    grid_svm = GridSearchCV(
        pipeline_svm,  # pipeline from above
        param_grid=param_svm,  # parameters to tune via cross validation
        refit=True,  # fit using all data, on the best detected classifier
        n_jobs=1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?
        cv=StratifiedKFold(n_splits=5),  # what type of cross validation to use
    )

    svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
    print('SVM '+info)


    print ('\nConfusion Matrix')
    print (confusion_matrix(label_test, svm_detector.predict(msg_test)))
    print (classification_report(label_test, svm_detector.predict(msg_test)))


    joblib.dump(svm_detector, 'tweetSent.pkl')




msg_trainp, msg_testp, label_trainp, label_testp = traintest(combi)
tonyFunction(msg_trainp,msg_testp, label_trainp, label_testp,' Sent')




