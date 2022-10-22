import gensim.downloader as api
import nltk
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neural_network
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# 3.1
# word2vec = api.load('word2vec-google-news-300')
#
#
# # 3.2 Extract the words from Reddit posts
#
#
# with open('./goemotions.json', 'r') as file:
#     reddit_df = pd.read_json(file)
#     reddit_df.columns = ['post', 'emotion', 'sentiment']
#     # lowercase every post
#     reddit_df['post'] = reddit_df['post'].str.lower()
#
#     number_of_tokens: int = 0
#     number_of_hits_per_post_list: list[int] = []
#     number_of_tokens_per_list: list[int] = []
#     average_embedding_vectors_list: list[list:[float]] = []
#
#     # iterate through every row of the dataframe
#     for index, row in reddit_df.iterrows():
#         # tokenize the post
#         tokens = nltk.word_tokenize(row['post'])
#         # add the number of tokens in the post to the total count
#         number_of_tokens += len(tokens)
#         number_of_tokens_per_list.append(len(tokens))
#         tokens_vector_list: list = []
#         number_of_hits: int = 0
#         # loop through the tokens
#         for token in tokens:
#             # try to get the token vector
#             try:
#                 vector = word2vec[token]
#                 number_of_hits += 1
#                 tokens_vector_list.append(vector)
#
#             except KeyError:
#                 continue
#         # add the number of hits to the list
#         number_of_hits_per_post_list.append(number_of_hits)
#
#        # calculate the average of the post
#         average_vector: list[float] = []
#         if len(tokens_vector_list) != 0:
#
#             for index in range(0, len(tokens_vector_list[0])):
#                 value_to_add: float = 0
#                 for token in tokens_vector_list:
#                     value_to_add += token[index]
#                 average_vector.append(value_to_add/len(tokens_vector_list))
#
#             average_embedding_vectors_list.append(average_vector)
#         else:
#             average_embedding_vectors_list.append([])
#
#
#
#     reddit_df['average_value'] = average_embedding_vectors_list
#     reddit_df['number_of_hits'] = number_of_hits_per_post_list
#     reddit_df['number_of_tokens'] = number_of_tokens_per_list
#
#     reddit_df.to_csv('output.csv', index=False, encoding='utf-8')

reddit_df = pd.read_csv('output.csv')
reddit_df_cleaned = reddit_df[reddit_df['number_of_hits'] > 0]

numpy_reddit_dt = reddit_df_cleaned.to_numpy()
labelEncoder = preprocessing.LabelEncoder()
X = numpy_reddit_dt[:, 3]  # features
Y = labelEncoder.fit_transform(numpy_reddit_dt[:, 1])  # emotions
Z = labelEncoder.fit_transform(numpy_reddit_dt[:, 2])  # sentiments
number_of_hits_array = numpy_reddit_dt[:, 4]
number_of_tokens_array = numpy_reddit_dt[:, 5]

x_train, x_test, y_train, y_test, z_train, z_test, number_of_hits_train, number_of_hits_test, number_of_tokens_train, number_of_tokens_test = train_test_split(X, Y, Z, number_of_hits_array, number_of_tokens_array, test_size=0.2, train_size=0.8, shuffle=False)

# calculate the percentage of hits in the train set
total_train_hits = 0
total_train_tokens = 0
for index, (hits, tokens) in enumerate(zip(number_of_hits_train, number_of_tokens_train)):
    total_train_tokens += tokens
    total_train_hits += hits

# calculate the percentage of hits in the test set
total_test_hits = 0
total_test_tokens = 0
for index, (hits, tokens) in enumerate(zip(number_of_hits_test, number_of_tokens_test)):
    total_test_tokens += tokens
    total_test_hits += hits

output_file = open('step3_output.txt', 'a')
emotion_mlp = neural_network.MLPClassifier()
emotion_mlp.fit(x_train, y_train)

sentiment_mlp = neural_network.MLPClassifier()
sentiment_mlp.fit(x_train, z_train)

emotion_mlp_prediction = emotion_mlp.predict(x_test)
sentiment_mlp_prediction = sentiment_mlp.predict(x_test)

emotion_mlp_matrix = confusion_matrix(y_test, emotion_mlp_prediction, labels=emotion_mlp.classes_)
sentiment_mlp_matrix = confusion_matrix(z_test, sentiment_mlp_prediction, labels=sentiment_mlp.classes_)

emotion_mlp_display = ConfusionMatrixDisplay(
    confusion_matrix=emotion_mlp_matrix,
    display_labels=emotion_mlp.classes_,
)
emotion_mlp_display.plot()
plt.show()

sentiment_mlp_display = ConfusionMatrixDisplay(
    confusion_matrix=sentiment_mlp_matrix,
    display_labels=sentiment_mlp.classes_,
)
sentiment_mlp_display.plot()
plt.show()

output_file.write('===================== CLASSIFICATION REPORT FOR BASE-MLP =====================\n')
output_file.write('EMOTIONS REPORT\n')
output_file.write(classification_report(y_test, emotion_mlp_prediction) + '\n\n')
output_file.write('SENTIMENT REPORT\n')
output_file.write(classification_report(z_test, sentiment_mlp_prediction) + '\n\n')
print(classification_report(y_test, emotion_mlp_prediction))
print(classification_report(z_test, sentiment_mlp_prediction))

print(reddit_df)
