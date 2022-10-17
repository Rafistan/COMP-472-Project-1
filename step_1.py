import json
from typing import Dict, List
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def readFile():
    posts = []
    sentiments = []
    emotions = []
    postsCategorizedBySentiments: Dict[str, List[str]] = {}
    postsCategorizedByEmotions: Dict[str, List[str]] = {}
    with open('./goemotions.json', 'r') as file:
        redditPosts = json.load(file)
        for redditPost in redditPosts:
            if redditPost[1] not in emotions:
                emotions.append(redditPost[1])
                postsCategorizedByEmotions[redditPost[1]] = [redditPost[0]]
            else:
                postsCategorizedByEmotions[redditPost[1]].append(redditPost[0])

            if redditPost[2] not in sentiments:
                sentiments.append(redditPost[2])
                postsCategorizedBySentiments[redditPost[2]] = [redditPost[0]]
            else:
                postsCategorizedBySentiments[redditPost[2]].append(redditPost[0])
        return [postsCategorizedBySentiments, postsCategorizedByEmotions]


[sentiments, emotions] = readFile()
sentimentKeys = sentiments.keys()
emotionKeys = emotions.keys()
plt.pie(np.array([len(attr) for attr in sentimentKeys]), labels=list(sentiments), autopct='%1.1f%%')
plt.savefig("part1.pdf", format="pdf", bbox_inches="tight") 
plt.show()
plt.pie(np.array([len(attr) for attr in emotionKeys]), labels=list(emotions), autopct='%1.1f%%')
plt.savefig("part2.pdf", format="pdf", bbox_inches="tight") 
plt.show()

#savefig saves to pdf 
