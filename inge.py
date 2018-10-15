import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def read_corpus(corpus_file, use_hyperp):
    train = pd.read_csv(corpus_file,
            compression='xz',
            sep='\t',
            encoding='utf-8',
            index_col=0).dropna()
    text = train.text
    if use_hyperp:
        label = train.hyperp
    else:
        label = train.bias

    return text, label

def main():
    X, Y = read_corpus('hyperp-training-grouped.csv.xz', use_hyperp=True)
    split_point = int(0.75*len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]

    pipeline = Pipeline([('vec', CountVectorizer()),
    ('clf', MultinomialNB())])

    model = pipeline.fit(Xtrain, Ytrain)

    y_pred = model.predict(Xtest)

    print("Accuracy: {0}\n".format(accuracy_score(y_pred, Ytest)))
    print(classification_report(y_pred, Ytest))


if __name__ == "__main__":
    main()
