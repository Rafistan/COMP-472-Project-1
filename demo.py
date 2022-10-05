import json
from typing import Dict, List


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
                postsCategorizedByEmotions[redditPost[2]].append(redditPost[0])

            if redditPost[2] not in sentiments:
                sentiments.append(redditPost[2])
                postsCategorizedBySentiments[redditPost[2]] = [redditPost[0]]
            else:
                postsCategorizedBySentiments[redditPost[2]].append(redditPost[0])


readFile()