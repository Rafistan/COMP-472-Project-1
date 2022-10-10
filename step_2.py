from sklearn.feature_extraction.text import CountVectorizer
import json

with open('./goemotions.json', 'r') as file:
    redditPosts = json.load(file)
    descriptions = [redditPost[0] for redditPost in redditPosts]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(descriptions)
    print(vectorizer.get_feature_names_out())
