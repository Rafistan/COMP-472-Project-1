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

    output_file = open('step2_output.txt', 'a')
    redditPosts = json.load(file)
    #descriptions = [redditPost[0] for redditPost in redditPosts]
    array = np.array(redditPosts)
    vectorizer = CountVectorizer()
    labelEncoder = preprocessing.LabelEncoder()
    X = vectorizer.fit_transform(array[:, 0]) #features 
    Y = labelEncoder.fit_transform(array[:, 1]) #emotions
    Z = labelEncoder.fit_transform(array[:, 2]) #sentiments
    number_of_tokens = len(X.toarray()[0])
    output_file.write(f'The total number of unique tokens vectorized is: {number_of_tokens}\n')
    print(number_of_tokens)

    #dikran try train_test_split, no idea if it worked 
    x_train, x_test = train_test_split(X, test_size=0.2, train_size=0.8, shuffle=False)
    y_train, y_test = train_test_split(Y, test_size=0.2, train_size=0.8, shuffle=False)
    z_train, z_test = train_test_split(Z, test_size=0.2, train_size=0.8, shuffle=False) 


    # #multionmial nb ====================================================================================================
    # emotion_mnb = MultinomialNB()
    # emotion_mnb.fit(x_train, y_train)
    #
    # sentiment_mnb = MultinomialNB()
    # sentiment_mnb.fit(x_train, z_train)
    #
    # emotion_mnb_prediction = emotion_mnb.predict(x_test)
    # sentiment_mnb_prediction = sentiment_mnb.predict(x_test)
    #
    # emotion_mnb_matrix = confusion_matrix(y_test, emotion_mnb_prediction, labels=emotion_mnb.classes_)
    # sentiment_mnb_matrix = confusion_matrix(z_test, sentiment_mnb_prediction, labels=sentiment_mnb.classes_)
    #
    # emotion_mnb_display = ConfusionMatrixDisplay(
    #     confusion_matrix=emotion_mnb_matrix,
    #     display_labels=emotion_mnb.classes_,
    # )
    # emotion_mnb_display.plot()
    # plt.show()
    #
    # sentiment_mnb_display = ConfusionMatrixDisplay(
    #     confusion_matrix=sentiment_mnb_matrix,
    #     display_labels=sentiment_mnb.classes_,
    # )
    # sentiment_mnb_display.plot()
    # plt.show()
    #
    # output_file.write('===================== CLASSIFICATION REPORT FOR BASE-MNB =====================\n')
    # output_file.write('EMOTIONS REPORT\n')
    # output_file.write(classification_report(y_test, emotion_mnb_prediction) + '\n\n')
    # output_file.write('SENTIMENT REPORT\n')
    # output_file.write(classification_report(z_test, sentiment_mnb_prediction) + '\n\n')
    # print(classification_report(y_test, emotion_mnb_prediction))
    # print(classification_report(z_test, sentiment_mnb_prediction))
    #
    # #decision tree =====================================================================================================
    # emotion_dtc = tree.DecisionTreeClassifier()
    # emotion_dtc.fit(x_train, y_train)
    #
    # sentiment_dtc = tree.DecisionTreeClassifier()
    # sentiment_dtc.fit(x_train, z_train)
    #
    # emotion_dtc_prediction = emotion_dtc.predict(x_test)
    # sentiment_dtc_prediction = sentiment_dtc.predict(x_test)
    #
    # emotion_dtc_matrix = confusion_matrix(y_test, emotion_dtc_prediction, labels=emotion_dtc.classes_)
    # sentiment_dtc_matrix = confusion_matrix(z_test, sentiment_dtc_prediction, labels=sentiment_dtc.classes_)
    #
    # emotion_dtc_display = ConfusionMatrixDisplay(
    #     confusion_matrix=emotion_dtc_matrix,
    #     display_labels=emotion_dtc.classes_,
    # )
    # emotion_dtc_display.plot()
    # plt.show()
    #
    # sentiment_dtc_display = ConfusionMatrixDisplay(
    #     confusion_matrix=sentiment_dtc_matrix,
    #     display_labels=sentiment_dtc.classes_,
    # )
    # sentiment_dtc_display.plot()
    # plt.show()
    #
    # output_file.write('===================== CLASSIFICATION REPORT FOR BASE-DT =====================\n')
    # output_file.write('EMOTIONS REPORT\n')
    # output_file.write(classification_report(y_test, emotion_dtc_prediction) + '\n\n')
    # output_file.write('SENTIMENT REPORT\n')
    # output_file.write(classification_report(z_test, sentiment_dtc_prediction) + '\n\n')
    # print(classification_report(y_test, emotion_dtc_prediction))
    # print(classification_report(z_test, sentiment_dtc_prediction))
    #
    # #multi layered perceptron ==========================================================================================
    # emotion_mlp = neural_network.MLPClassifier()
    # emotion_mlp.fit(x_train, y_train)
    #
    # sentiment_mlp = neural_network.MLPClassifier()
    # sentiment_mlp.fit(x_train, z_train)
    #
    # emotion_mlp_prediction = emotion_mlp.predict(x_test)
    # sentiment_mlp_prediction = sentiment_mlp.predict(x_test)
    #
    # emotion_mlp_matrix = confusion_matrix(y_test, emotion_mlp_prediction, labels=emotion_mlp.classes_)
    # sentiment_mlp_matrix = confusion_matrix(z_test, sentiment_mlp_prediction, labels=sentiment_mlp.classes_)
    #
    # emotion_mlp_display = ConfusionMatrixDisplay(
    #     confusion_matrix=emotion_mlp_matrix,
    #     display_labels=emotion_mlp.classes_,
    # )
    # emotion_mlp_display.plot()
    # plt.show()
    #
    # sentiment_mlp_display = ConfusionMatrixDisplay(
    #     confusion_matrix=sentiment_mlp_matrix,
    #     display_labels=sentiment_mlp.classes_,
    # )
    # sentiment_mlp_display.plot()
    # plt.show()
    #
    # output_file.write('===================== CLASSIFICATION REPORT FOR BASE-MLP =====================\n')
    # output_file.write('EMOTIONS REPORT\n')
    # output_file.write(classification_report(y_test, emotion_mlp_prediction) + '\n\n')
    # output_file.write('SENTIMENT REPORT\n')
    # output_file.write(classification_report(z_test, sentiment_mlp_prediction) + '\n\n')
    # print(classification_report(y_test, emotion_mlp_prediction))
    # print(classification_report(z_test, sentiment_mlp_prediction))
    #
    # #gridsearchCV multinomail naive bayes ==============================================================================
    # # https://www.kaggle.com/code/sfktrkl/titanic-hyperparameter-tuning-gridsearchcv
    # hyperparameters = {'alpha': [0, 0.5, 1.0, 1.5]}
    #
    # emotion_grid_mnb = GridSearchCV(MultinomialNB(), hyperparameters, refit=True, verbose=3, n_jobs=-1)
    # emotion_grid_mnb.fit(x_train, y_train)
    #
    # sentiment_grid_mnb = GridSearchCV(MultinomialNB(), hyperparameters, refit=True, verbose=3, n_jobs=-1)
    # sentiment_grid_mnb.fit(x_train, z_train)
    #
    # emotion_gird_mnb_prediction = emotion_grid_mnb.predict(x_test)
    # sentiment_gird_mnb_prediction = sentiment_grid_mnb.predict(x_test)
    #
    # emotion_grid_mnb_matrix = confusion_matrix(y_test, emotion_gird_mnb_prediction, labels=emotion_grid_mnb.classes_)
    # sentiment_grid_mnb_matrix = confusion_matrix(z_test, sentiment_gird_mnb_prediction, labels=sentiment_grid_mnb.classes_)
    #
    # emotion_grid_mnb_display = ConfusionMatrixDisplay(
    #     confusion_matrix=emotion_grid_mnb_matrix,
    #     display_labels=emotion_grid_mnb.classes_,
    # )
    # emotion_grid_mnb_display.plot()
    # plt.show()
    #
    # sentiment_grid_mnb_display = ConfusionMatrixDisplay(
    #     confusion_matrix=sentiment_grid_mnb_matrix,
    #     display_labels=sentiment_grid_mnb.classes_,
    # )
    # sentiment_grid_mnb_display.plot()
    # plt.show()
    #
    # output_file.write('===================== CLASSIFICATION REPORT FOR TOP-MNB =====================\n')
    # output_file.write('EMOTIONS REPORT\n')
    # output_file.write(classification_report(y_test, emotion_gird_mnb_prediction) + '\n\n')
    # output_file.write('SENTIMENT REPORT\n')
    # output_file.write(classification_report(z_test, sentiment_gird_mnb_prediction) + '\n\n')
    # print(classification_report(y_test, emotion_gird_mnb_prediction))
    # print(classification_report(z_test, sentiment_gird_mnb_prediction))

    #gridsearchCV decision tree ========================================================================================
    hyperparameters = {'criterion': ['gini', 'entropy'],
                'max_depth': [2, 3],
                'min_samples_split': [2, 3, 4]}

    emotion_grid_dt = GridSearchCV(tree.DecisionTreeClassifier(), hyperparameters, refit=True, verbose=3, n_jobs=-1)
    emotion_grid_dt.fit(x_train, y_train)

    sentiment_grid_dt = GridSearchCV(tree.DecisionTreeClassifier(), hyperparameters, refit=True, verbose=3, n_jobs=-1)
    sentiment_grid_dt.fit(x_train, z_train)

    emotion_gird_dt_prediction = emotion_grid_dt.predict(x_test)
    sentiment_gird_dt_prediction = sentiment_grid_dt.predict(x_test)

    emotion_grid_dt_matrix = confusion_matrix(y_test, emotion_gird_dt_prediction, labels=emotion_grid_dt.classes_)
    sentiment_grid_dt_matrix = confusion_matrix(z_test, sentiment_gird_dt_prediction, labels=sentiment_grid_dt.classes_)

    emotion_grid_dt_display = ConfusionMatrixDisplay(
        confusion_matrix=emotion_grid_dt_matrix,
        display_labels=emotion_grid_dt.classes_,
    )
    emotion_grid_dt_display.plot()
    plt.show()

    sentiment_grid_dt_display = ConfusionMatrixDisplay(
        confusion_matrix=sentiment_grid_dt_matrix,
        display_labels=sentiment_grid_dt.classes_,
    )
    sentiment_grid_dt_display.plot()
    plt.show()

    output_file.write('===================== CLASSIFICATION REPORT FOR TOP-DT =====================\n')
    output_file.write('EMOTIONS REPORT\n')
    output_file.write(classification_report(y_test, emotion_gird_dt_prediction)+ '\n\n')
    output_file.write('SENTIMENT REPORT\n')
    output_file.write(classification_report(z_test, sentiment_gird_dt_prediction)+ '\n\n')
    print(classification_report(y_test, emotion_gird_dt_prediction))
    print(classification_report(z_test, sentiment_gird_dt_prediction))

    #gridsearchCV top MLP ==============================================================================================
    #parameters might be wrong

    hyperparameters = {'hidden_layer_sizes': [(30,), (50,)],
                       'activation': ['tanh', 'relu', 'sigmoid', 'identity'],
                       'solver': ['adam', 'stochastic']}

    emotion_grid_mlp = GridSearchCV(neural_network.MLPClassifier(), hyperparameters, refit=True)
    emotion_grid_mlp.fit(x_train, y_train)

    sentiment_grid_mlp = GridSearchCV(neural_network.MLPClassifier(), hyperparameters, refit=True,verbose=3, n_jobs=-1)
    sentiment_grid_mlp.fit(x_train, z_train)

    emotion_grid_mlp_prediction = emotion_grid_mlp.predict(x_test)
    sentiment_grid_mlp_prediction = sentiment_grid_mlp.predict(x_test)

    emotion_grid_mlp_matrix = confusion_matrix(y_test, emotion_grid_mlp_prediction, labels=emotion_grid_mlp.classes_)
    sentiment_grid_mlp_matrix = confusion_matrix(z_test, sentiment_grid_mlp_prediction, labels=sentiment_grid_mlp.classes_)

    emotion_grid_mlp_display = ConfusionMatrixDisplay(
        confusion_matrix=emotion_grid_mlp_matrix,
        display_labels=emotion_grid_mlp.classes_,
    )
    emotion_grid_mlp_display.plot()
    plt.show()

    sentiment_grid_mlp_display = ConfusionMatrixDisplay(
        confusion_matrix=sentiment_grid_mlp_matrix,
        display_labels=sentiment_grid_mlp.classes_,
    )
    sentiment_grid_mlp_display.plot()
    plt.show()

    output_file.write('===================== CLASSIFICATION REPORT FOR TOP-MLP =====================\n')
    output_file.write('EMOTIONS REPORT\n')
    output_file.write(classification_report(y_test, emotion_grid_mlp_prediction) + '\n\n')
    output_file.write('SENTIMENT REPORT\n')
    output_file.write(classification_report(z_test, sentiment_grid_mlp_prediction) + '\n\n')
    output_file.close()
    print(classification_report(y_test, emotion_grid_mlp_prediction))
    print(classification_report(z_test, sentiment_grid_mlp_prediction))

    





 

