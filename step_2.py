from sklearn.feature_extraction.text import CountVectorizer
import json
from sklearn.model_selection import train_test_split
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
    LabelEncoder = preprocessing.LabelEncoder()
    X = vectorizer.fit_transform(array[:, 0]) #features 
    Y = LabelEncoder.fit_transform(array[:, 1]) #emotions 
    Z = LabelEncoder.fit_transform(array[:, 2]) #sentiments
    print(vectorizer.get_feature_names_out().size)

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
    emotionclf = MultinomialNB()
    emotionclf.fit(x_train, y_train)

    sentimentclf = MultinomialNB() 
    sentimentclf.fit(x_train, z_train) 

    emotionclfprd = emotionclf.predict(x_test)
    sentimentclfprd = sentimentclf.predict(x_test)

    print(emotionclfprd) 
    print(sentimentclfprd)
    print(confusion_matrix(y_test, emotionclfprd)) 
    print(confusion_matrix(z_test, sentimentclfprd))
    print(classification_report(y_test, emotionclfprd)) 
    print(classification_report(z_test, sentimentclfprd)) 

    #decision tree
    emotiondtc = tree.DecisionTreeClassifier()
    emotiondtc.fit(x_train, y_train)

    sentimentdtc = tree.DecisionTreeClassifier()
    sentimentdtc.fit(x_train, z_train) 

    emotiondtcprd = emotiondtc.predict(x_test)
    sentimentdtcprd = sentimentclf.predict(x_test) 

    print(emotiondtcprd) 
    print(sentimentdtcprd)
    print(confusion_matrix(y_test, emotiondtcprd)) 
    print(confusion_matrix(z_test, sentimentdtcprd))
    print(classification_report(y_test, emotiondtcprd)) 
    print(classification_report(z_test, sentimentdtcprd)) 
    
    #multi layered perceptron
    emotionmlp = neural_network.MLPClassifier() 
    emotionmlp.fit(x_train, y_train)

    sentimentmlp = neural_network.MLPClassifier() 
    sentimentmlp.fit(x_train, z_train)

    emotionmlpprd = emotionmlp.predict(x_test)
    sentimentmlpprd = sentimentmlp.predict(x_test) 

    print(emotionmlpprd) 
    print(sentimentmlpprd)
    print(confusion_matrix(y_test, emotionmlpprd)) 
    print(confusion_matrix(z_test, sentimentmlpprd))
    print(classification_report(y_test, emotionmlpprd)) 
    print(classification_report(z_test, sentimentmlpprd)) 

    #gridsearchCV

    parameters = {'alpha': [0,0.5,1.0,1.5]}

    emotioncvgridclf = GridSearchCV(MultinomialNB(),parameters,refit=True,verbose=3,n_jobs=-1)  
    emotioncvgridclf.fit(x_train, y_train) 

    sentimentcvgridclf = GridSearchCV(MultinomialNB(),parameters,refit=True,verbose=3, n_jobs=-1)
    sentimentcvgridclf.fit(x_train, z_train)

    emotioncvprd = emotioncvgridclf.predict(x_test)
    sentimentcvprd = sentimentcvgridclf.predict(x_test)

    print(emotioncvprd) 
    print(sentimentcvprd)
    print(confusion_matrix(y_test, emotioncvprd)) 
    print(confusion_matrix(z_test, sentimentcvprd))
    print(classification_report(y_test, emotioncvprd)) 
    print(classification_report(z_test, sentimentcvprd)) 

    #gridsearchCV decision tree

    paramsdt = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4], 'min_samples_split': [2,3,4]} 

    emotioncvgriddt = GridSearchCV(tree.DecisionTreeClassifier(),paramsdt,refit=True,verbose=3, n_jobs=-1) 
    emotioncvgriddt.fit(x_train, y_train) 

    sentimentcvgriddt = GridSearchCV(tree.DecisionTreeClassifier(),paramsdt,refit=True,verbose=3, n_jobs=-1)
    sentimentcvgriddt.fit(x_train, z_train)

    emotioncvprddt = emotioncvgriddt.predict(x_test)
    sentimentcvprddt = sentimentcvgriddt.predict(x_test)

    print(emotioncvprddt) 
    print(sentimentcvprddt)
    print(confusion_matrix(y_test, emotioncvprddt)) 
    print(confusion_matrix(z_test, sentimentcvprddt))
    print(classification_report(y_test, emotioncvprddt)) 
    print(classification_report(z_test, sentimentcvprddt))


    #gridsearchCV top MLP 
    





 

