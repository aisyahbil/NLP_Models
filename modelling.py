import numpy as np
import pandas as pd
#from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn import preprocessing
#from sklearn.externals import joblib
from sklearn.model_selection import learning_curve
import pickle
from joblib import dump, load


def holdout(data, test_size = 0.2):
    train, test = train_test_split(data, test_size=test_size, random_state=1)
    X_train = train['cleaned_text'].values
    X_test = test['cleaned_text'].values
    y_train = train['Label'].values
    y_test = test['Label'].values
    return X_train, X_test, y_train, y_test 
    
def crossvaldata(data):
    X = data['cleaned_text'].values
    y = data['Label'].values
    return X,y

def vektorisasi(data, termfrequency = True): #data yang sudah di clean
    if termfrequency:   
        tf_vectorizer = CountVectorizer(max_df=1.0, min_df=1) 
        dtm_tf = tf_vectorizer.fit_transform(data)
        tf_terms = tf_vectorizer.get_feature_names() 
        #print(dtm_tf.shape)
    else:
        tf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1) 
        dtm_tf = tf_vectorizer.fit_transform(data)
        tf_terms = tf_vectorizer.get_feature_names() 
        #print(dtm_tf.shape)
    return tf_vectorizer,dtm_tf,tf_terms


def savetokenizer(filepath, tokenizer):
    with open(filepath, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#def savevektorisasi(data, filename):
#    with open('vectorizer.pk', 'wb') as fin:
#        pickle.dump(vectorizer, fin)

def matrixtermfreq(data, termfrequency = True):
    #myvocabulary = tf_terms
    corpus = {i : str(data['cleaned_text'].iloc[i]) for i in range(0,len(data))}
    if termfrequency:
        tf = CountVectorizer(max_df=1, min_df=1)
        tfs = tf.fit_transform(corpus.values())
        feature_names = tf.get_feature_names()
    else:
        tfidf = TfidfVectorizer(max_df=1, min_df=1)
        tfs = tfidf.fit_transform(corpus.values())
        feature_names = tfidf.get_feature_names()
    
    corpus_index = [n for n in corpus]
    df = pd.DataFrame(tfs.T.todense(), index=feature_names, columns=corpus_index)
    #print(df)
    return df

def modelling(data, modelname = str(), crossval = True,  termfrequency = False, n_fold = 3, kernel = 'linear', n_jobs=1):
    data.cleaned_text = data.cleaned_text.astype('str')
    vektor = vektorisasi(data.cleaned_text, termfrequency = termfrequency)
    savetokenizer('./NLP_Models/tokenizer_hatespeech/'+ modelname+ '_tokenizer.pickle', vektor[0])
    kfolds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1)
    np.random.seed(1)
    pipeline_svm = make_pipeline(vektor[0], 
                                SVC(probability=True, kernel=kernel, class_weight="balanced"))
    
    grid_svm = GridSearchCV(pipeline_svm,
                        param_grid = {'svc__C': [0.01, 0.1]}, 
                        cv = kfolds,
                        scoring="roc_auc",
                        verbose=1,   
                        n_jobs=n_jobs) 
    if crossval:
        X, y = crossvaldata(data)
        #grid_svm.fit(X, y)
        model = grid_svm.fit(X, y)
        score = grid_svm.score(X, y)
        print("roc_auc model terbaik adalah:", score)
        scorebest = grid_svm.best_estimator_.score(X,y)
        print("roc_auc model estimator terbaik adalah:", scorebest)
        bestparameter =  grid_svm.best_params_
        print("Parameter terbaik adalah:", bestparameter)
        bestscore = grid_svm.best_score_
        print("Rataan roc_auc model tiap fold adalah:" ,bestscore)
        filename = './NLP_Models/model_hatespeech/'+ modelname + 'hate_detection.joblib'
        dump(model, filename, compress = 1)
    else:
        X_train, X_test, y_train, y_test  = holdout(data)
        #grid_svm.fit(X_train, y_train)
        model = grid_svm.fit(X_train, y_train)
        score = grid_svm.score(X_test, y_test)
        print("roc_auc model terbaik adalah:",score)
        scorebest = grid_svm.best_estimator_.score(X_test,y_test)
        print("roc_auc model estimator terbaik adalah:", scorebest)
        bestparameter =  grid_svm.best_params_
        print("Parameter terbaik adalah:", bestparameter)
        bestscore = grid_svm.best_score_
        print("Rataan roc_auc model tiap fold adalah:" ,bestscore)
        filename = './NLP_Models/model_hatespeech/'+ modelname + 'hate_detection.joblib'
        dump(model, filename)
    return model, score, bestscore


def confusionMatrix(model, X,y):
    y_pred = model.best_estimator_.predict(X)
    cm = confusion_matrix(y, y_pred)
    return cm

def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        

    auC = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, Labels=["-1"], average='micro', pos_Label="-1")
    prec = precision_score(y, pred, Labels=["-1"], average='micro', pos_Label="-1")
    rec = recall_score(y, pred, Labels=["-1"], average='micro', pos_Label="-1")
   
    return {'auc': auC, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}

def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr

def visualroc(model,X,y):    
    roc_svm = get_roc_curve(model, X, y)
    
    fpr, tpr = roc_svm
    plt.figure(figsize=(14,8))
    plt.plot(fpr, tpr, color="red")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve')
    plt.show()

#train_sizes, train_scores, test_scores = \
#    learning_curve(model.best_estimator_, X_train, y_train, cv=10, n_jobs=-1, 
#                   scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)

def loadmodel(filename):
    model = load(filename)
    return model

def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14,8)):

    plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="lower right")
#    return plt

def showlearningcurve( X, y, train_sizes, train_scores, test_scores):

    plot_learning_curve(X, y, train_sizes, train_scores, test_scores, ylim=(0.7, 1.01), figsize=(14,6))
    plt.show()
    
#    return train_sizes, train_scores, test_scores