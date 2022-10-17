from sklearn.feature_extraction.text import CountVectorizer
import json
from sklearn.model_selection import train_test_split 

with open('./goemotions.json', 'r') as file:
    redditPosts = json.load(file)
    descriptions = [redditPost[0] for redditPost in redditPosts]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(descriptions)
    print(vectorizer.get_feature_names_out())

    #dikran try train_test_split, no idea if it worked 
    x_train, x_test = train_test_split(X, test_size=0.2)
    print('xtrain', x_train) 
    print('x_test', x_test) 
