import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd

# 3.1
word2vec = api.load('word2vec-google-news-300')


# 3.2 Extract the words from Reddit posts


with open('./goemotions.json', 'r') as file:
    reddit_df = pd.read_json(file)
    reddit_df.columns = ['post', 'emotion', 'sentiment']
    # lowercase every post
    reddit_df['post'] = reddit_df['post'].str.lower()

    number_of_tokens: int = 0
    number_of_hits: int = 0
    number_of_misses: int = 0
    number_of_hits_per_post_list: list[int] = []
    average_of_embeddings: list[float] = []
    # iterate through every row of the dataframe
    for index, row in reddit_df.iterrows():
        # tokenize the post
        tokens = nltk.word_tokenize(row['post'])
        # add the number of tokens in the post to the total count
        number_of_tokens += len(tokens)
        average_tokens_value_list: list[float] = []
        # loop through the tokens
        for token in tokens:
            # try to get the token vector
            try:
                vector = word2vec[token]
                number_of_hits += 1
                average_value_of_token: float = 0

                # add all the vector values together
                for value in vector:
                    average_value_of_token += value

                # average the token's value and add it to the list
                average_tokens_value_list.append(average_value_of_token/len(vector))

            except KeyError:
                # add one to the number of misses
                number_of_misses += 1
        # after all the post's tokens have been calculated a value, average it again
        average_number_post: float = 0

        for token_value in average_tokens_value_list:
            average_number_post += token_value
        # add the average of the post to the list
        # check that the post has at least one hit
        if len(average_tokens_value_list) == 0:
            average_of_embeddings.append(0)
        else:
            average_of_embeddings.append(average_number_post/len(average_tokens_value_list))

    reddit_df['average_value'] = average_of_embeddings
    reddit_df.to_csv('output.csv', index=False, encoding='utf-8')


    print(reddit_df)
