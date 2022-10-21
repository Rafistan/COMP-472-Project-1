from sklearn.feature_extraction.text import CountVectorizer
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree, neural_network 
from sklearn.model_selection import GridSearchCV 


from sklearn.naive_bayes import MultinomialNB 

with open('./goemotions.json', 'r') as file:
    redditPosts = json.load(file)
    #descriptions = [redditPost[0] for redditPost in redditPosts]
    array = np.array(redditPosts)
    vectorizer = CountVectorizer()
    labelEncoder = preprocessing.LabelEncoder()
    X = vectorizer.fit_transform(array[:, 0]) #features 
    Y = labelEncoder.fit_transform(array[:, 1]) #emotions
    Z = labelEncoder.fit_transform(array[:, 2]) #sentiments
    number_of_tokens = len(X.toarray()[0])
    print(number_of_tokens)

    #dikran try train_test_split, no idea if it worked 
    x_train, x_test = train_test_split(X, test_size=0.2, train_size=0.8, shuffle=False)
    y_train, y_test = train_test_split(Y, test_size=0.2, train_size=0.8, shuffle=False)
    z_train, z_test = train_test_split(Z, test_size=0.2, train_size=0.8, shuffle=False) 
    #print('X') 
    #print('x_train', x_train) 
    #print('x_test', x_test)
    #print('Y')
    #print('y_train', y_train)
    #print('y_test', y_test)
    #print('Z')
    #print('z_train', z_train)
    #print('z_test', z_test) 

    #multionmial nb 
    emotion_mnb = MultinomialNB()
    emotion_mnb.fit(x_train, y_train)

    sentiment_mnb = MultinomialNB()
    sentiment_mnb.fit(x_train, z_train)

    emotion_mnb_prediction = emotion_mnb.predict(x_test)
    sentiment_mnb_prediction = sentiment_mnb.predict(x_test)

    emotion_mnb_matrix = confusion_matrix(y_test, emotion_mnb_prediction, labels=emotion_mnb.classes_)
    sentiment_mnb_matrix = confusion_matrix(z_test, sentiment_mnb_prediction, labels=sentiment_mnb.classes_)

    emotion_mnb_display = ConfusionMatrixDisplay(
        confusion_matrix=emotion_mnb_matrix,
        display_labels=emotion_mnb.classes_,
    )
    emotion_mnb_display.plot()
    plt.show()
    plt.savefig("emotion_mnb.pdf", format="pdf")

    sentiment_mnb_display = ConfusionMatrixDisplay(
        confusion_matrix=sentiment_mnb_matrix,
        display_labels=sentiment_mnb.classes_,
    )
    sentiment_mnb_display.plot()
    plt.show()
    plt.savefig("sentiment_mnb.pdf", format="pdf")

    print(emotion_mnb_prediction)
    print(sentiment_mnb_prediction)
    print(emotion_mnb_matrix)
    print(sentiment_mnb_matrix)
    print(classification_report(y_test, emotion_mnb_prediction))
    print(classification_report(z_test, sentiment_mnb_prediction))

    #decision tree
    emotion_dtc = tree.DecisionTreeClassifier()
    emotion_dtc.fit(x_train, y_train)

    sentiment_dtc = tree.DecisionTreeClassifier()
    sentiment_dtc.fit(x_train, z_train)

    emotion_dtc_prediction = emotion_dtc.predict(x_test)
    sentiment_dtc_prediction = sentiment_dtc.predict(x_test)

    emotion_dtc_matrix = confusion_matrix(y_test, emotion_dtc_prediction, labels=emotion_dtc.classes_)
    sentiment_dtc_matrix = confusion_matrix(z_test, sentiment_dtc_prediction, labels=sentiment_dtc.classes_)

    emotion_dtc_display = ConfusionMatrixDisplay(
        confusion_matrix=emotion_dtc_matrix,
        display_labels=emotion_dtc.classes_,
    )
    emotion_dtc_display.plot()
    plt.show()
    plt.savefig("emotion_dtc.pdf", format="pdf", bbox_inches="tight")

    sentiment_dtc_display = ConfusionMatrixDisplay(
        confusion_matrix=sentiment_dtc_matrix,
        display_labels=sentiment_dtc.classes_,
    )
    sentiment_dtc_display.plot()
    plt.show()
    plt.savefig("sentiment_dtc.pdf", format="pdf", bbox_inches="tight")

    print(emotion_dtc_prediction)
    print(sentiment_dtc_prediction)
    print(emotion_dtc_matrix)
    print(sentiment_dtc_matrix)
    print(classification_report(y_test, emotion_dtc_prediction))
    print(classification_report(z_test, sentiment_dtc_prediction))
    
    #multi layered perceptron
    emotionmlp = neural_network.MLPClassifier() 
    emotionmlp.fit(x_train, y_train)

    sentimentmlp = neural_network.MLPClassifier() 
    sentimentmlp.fit(x_train, z_train)

    emotionmlpprd = emotionmlp.predict(x_test)
    sentimentmlpprd = sentimentmlp.predict(x_test) 

    emotion_mlp_matrix = confusion_matrix(y_test, emotionmlpprd, labels=emotionmlp.classes_)
    sentiment_mlp_matrix = confusion_matrix(z_test, sentimentmlpprd, labels=sentimentmlp.classes_)

    emotion_display = ConfusionMatrixDisplay(
        confusion_matrix=emotion_mlp_matrix,
        display_labels=emotionmlp.classes_,
    )
    emotion_display.plot()
    plt.show()
    plt.savefig("emotion_mlp.pdf", format="pdf", bbox_inches="tight")

    sentiment_display = ConfusionMatrixDisplay(
        confusion_matrix=sentiment_mlp_matrix,
        display_labels=sentimentmlp.classes_,
    )
    sentiment_display.plot()
    plt.show()
    plt.savefig("sentiment_mlp.pdf", format="pdf", bbox_inches="tight")

    print(emotionmlpprd) 
    print(sentimentmlpprd)
    print(emotion_mlp_matrix)
    print(sentiment_mlp_matrix)
    print(classification_report(y_test, emotionmlpprd)) 
    print(classification_report(z_test, sentimentmlpprd)) 

    # #gridsearchCV multinomail naive bayes
    #
    # paramsmb = {'alpha': [0,0.5,1.0,1.5]}
    #
    # emotioncvgridclf = GridSearchCV(MultinomialNB(),paramsmb,refit=True,verbose=3,n_jobs=-1)
    # emotioncvgridclf.fit(x_train, y_train)
    #
    # sentimentcvgridclf = GridSearchCV(MultinomialNB(),paramsmb,refit=True,verbose=3, n_jobs=-1)
    # sentimentcvgridclf.fit(x_train, z_train)
    #
    # emotioncvprd = emotioncvgridclf.predict(x_test)
    # sentimentcvprd = sentimentcvgridclf.predict(x_test)
    #
    # print(emotioncvprd)
    # print(sentimentcvprd)
    # print(confusion_matrix(y_test, emotioncvprd))
    # print(confusion_matrix(z_test, sentimentcvprd))
    # print(classification_report(y_test, emotioncvprd))
    # print(classification_report(z_test, sentimentcvprd))
    #
    # #gridsearchCV decision tree
    # #parameters might be wrong
    #
    # paramsdt = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4], 'min_samples_split': [2,3,4]}
    #
    # emotioncvgriddt = GridSearchCV(tree.DecisionTreeClassifier(),paramsdt,refit=True,verbose=3, n_jobs=-1)
    # emotioncvgriddt.fit(x_train, y_train)
    #
    # sentimentcvgriddt = GridSearchCV(tree.DecisionTreeClassifier(),paramsdt,refit=True,verbose=3, n_jobs=-1)
    # sentimentcvgriddt.fit(x_train, z_train)
    #
    # emotioncvprddt = emotioncvgriddt.predict(x_test)
    # sentimentcvprddt = sentimentcvgriddt.predict(x_test)
    #
    # print(emotioncvprddt)
    # print(sentimentcvprddt)
    # print(confusion_matrix(y_test, emotioncvprddt))
    # print(confusion_matrix(z_test, sentimentcvprddt))
    # print(classification_report(y_test, emotioncvprddt))
    # print(classification_report(z_test, sentimentcvprddt))
    #
    #
    # #gridsearchCV top MLP
    # #parameters might be wrong
    #
    # paramsmlp = {'hidden_layer_size': [(30,),(50,)], 'activation': ['tanh', 'relu', 'sigmoid', 'identity'], 'solver': ['adam', 'stochastic']}
    #
    # emotioncvgridmlp = GridSearchCV(neural_network.MLPClassifier(),paramsmlp, refit=True)
    # emotioncvgridmlp.fit(x_train, y_train)
    #
    # sentimentcvgridmlp = GridSearchCV(neural_network.MLPClassifier(),paramsmlp,refit=True,verbose=3, n_jobs=-1)
    # sentimentcvgridmlp.fit(x_train, z_train)
    #
    # emotioncvprdmlp = emotioncvgridmlp.predict(x_test)
    # sentimentcvprdmlp = sentimentcvgridmlp.predict(x_test)
    #
    # print(emotioncvprdmlp)
    # print(sentimentcvprdmlp)
    # print(confusion_matrix(y_test, emotioncvprdmlp))
    # print(confusion_matrix(z_test, sentimentcvprdmlp))
    # print(classification_report(y_test, emotioncvprdmlp))
    # print(classification_report(z_test, sentimentcvprdmlp))
    





 

