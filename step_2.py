from sklearn.feature_extraction.text import CountVectorizer
import json
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn import preprocessing

with open('./goemotions.json', 'r') as file:
    redditPosts = json.load(file)
    #descriptions = [redditPost[0] for redditPost in redditPosts]
    array = np.array(redditPosts) 
    vectorizer = CountVectorizer()
    labEncoder = preprocessing.LabelEncoder()
    X = vectorizer.fit_transform(array[:, 0]) #features 
    Y = labEncoder.fit_transform(array[:, 1]) #emotions 
    Z = labEncoder.fit_transform(array[:, 2]) #sentiments
    #print(vectorizer.get_feature_names_out())

    #dikran try train_test_split, no idea if it worked 
    x_train, x_test = train_test_split(X, test_size=0.2, train_size=0.8, shuffle=False)
    y_train, y_test = train_test_split(Y, test_size=0.2, train_size=0.8, shuffle=False)
    z_train, z_test = train_test_split(Z, test_size=0.2, train_size=0.8, shuffle=False) 
    print('X') 
    print('x_train', x_train) 
    print('x_test', x_test)
    print('Y')
    print('y_train', y_train)
    print('y_test', y_test)
    print('Z')
    print('z_train', z_train)
    print('z_test', z_test) 
