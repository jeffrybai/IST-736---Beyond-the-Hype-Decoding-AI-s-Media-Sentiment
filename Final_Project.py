import os
import pandas as pd
import numpy as np
from datetime import datetime

from project_functions import *

from datetime import datetime, timedelta

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import tree

import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print('Instantiation: Vectorizers')
# Instantiation of CountVectorizer and Term Frequency - Inverse Document Frequency
cv = CountVectorizer(input='content',
                     stop_words='english',
                     tokenizer=stemmer_tokenizer,
                     #max_features=10000,
                     #ngram_range=(2,2),
                     lowercase=True)
tfidf = TfidfVectorizer(input='content',
                        stop_words='english',
                        tokenizer=stemmer_tokenizer,
                        #max_features=10000,
                        #ngram_range=(2,2),
                        lowercase=True)

n_gram = "Unigram"

print('Instantiation: Models')
# Instantiation of models: Multinomial Naive Bayes, Decision Tree Classifier, Support Vector Machine
mnb = MultinomialNB()
dtc = DecisionTreeClassifier()
svm = SVC()
lda_model = LatentDirichletAllocation(n_components=5, random_state=736)


print('Load CSV')
df_bias = pd.read_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\AllSides_Bias_Rating.csv', header=0)

#
#
# Getting Data
#
#

# Getting data from GDELT from 2017 to 2023
for year in range(2017, 2024):
    start_date = datetime(2023, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime(2023, 12, 31).strftime("%Y-%m-%d")
    
    
    for domain in df_bias["Domain"]:
        print(f"[{domain}] => {start_date} - {end_date}")
    
        try:
            df_temp = gdelt_query("artificial intelligence",
                                  start_date=start_date,
                                  end_date=end_date,
                                  domain=domain)
            df_temp['domain_key'] = domain
            df_temp.to_csv(f'C:\\Users\\Jeffry J. Bai\\Desktop\\IST 736\\gdelt\\{domain.split(".")[0]}url{year}.csv', index=False)
        except Exception as e:
            print(e)

folder_path = "C:\\Users\\Jeffry J. Bai\\Desktop\\IST 736\\gdelt"

df_time = pd.DataFrame(columns=['url', 'url_mobile', 'title', 'seendate', 'socialimage', 'domain', 'language', 'sourcecountry', 'domain_key'])

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Check if the file is a .txt file
        file_path = os.path.join(folder_path, file_name)
        df_temp = pd.read_csv(file_path, header=0)

    print(file_name)
    # Append the file name and content to the data frame
    df_time = df_time._append(df_temp, ignore_index=False)

text_list = []

for url in df_time['url']:
    print(url)
    try:
        text = get_article_text(url)
    except:
        text = ''
    finally:
        text_list.append(text)

df_time['text'] = text_list

df_time.to_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_gdelt_output.csv', index=False)


print('Setup')
## Setup for news API scrap
query = 'artificial intelligence'
key = '317dbc42bfff43cd981b49f680a49b83'

df = pd.DataFrame(columns=['Title', 'Source', 'Author', 'Summary', 'Date', 'Text', 'Title_nltk', 'Title_vader', 'Summary_nltk', 'Summary_vader', 'Text_nltk', 'Text_vader', 'Domain', 'Country'])

for domain in df_bias["Domain"]:
    df_temp = get_news_dataframe(query, key, domains=domain)
    df_temp['Domain'] = domain
    df_temp['Country'] = 'US'

    df_temp.to_csv(f'C:\\Users\\Jeffry J. Bai\\Desktop\\IST 736\\{domain.split(".")[0]}_output.csv', index=False)


df = pd.DataFrame(columns=['Title', 'Source', 'Author', 'Summary', 'Date', 'Text', 'Domain', 'Country'])

folder_path = "C:\\Users\\Jeffry J. Bai\\Desktop\\IST 736\\newsapi"

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Check if the file is a .txt file
        file_path = os.path.join(folder_path, file_name)
        df_temp = pd.read_csv(file_path, header=0)

    # Append the file name and content to the data frame
    df = df._append(df_temp, ignore_index=False)

df = df.merge(df_bias, on="Domain", how="left")

df['Title'] = df['Title'].astype(str)
df['Text'] = df['Text'].astype(str)
df['Summary'] = df['Summary'].astype(str)

## Labeling data
# Parts of speech tagging
df['POS_tag'] = [return_pos_tag(text) for text in df['Text']]

# Sentiment
df['Title_nltk'] = get_nltk_sentiment(df['Title'])
df['Title_vader'] = get_vader_sentiment(df['Title'])
df['Summary_nltk'] = get_nltk_sentiment(df['Summary'])
df['Summary_vader'] = get_vader_sentiment(df['Summary'])
df['Text_nltk'] = get_nltk_sentiment(df['Text'])
df['Text_vader'] = get_vader_sentiment(df['Text'])

df.to_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_newsapi_output.csv', index=False)

print('newsapi')
df_newsapi = pd.read_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_newsapi_output.csv', header=0)
df_newsapi['POS_tag'] = df_newsapi['POS_tag'].str.strip()
df_newsapi = df_newsapi.dropna(subset=['POS_tag'])

df_newsapi['Date'] = [datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ').date() for date in df_newsapi['Date']]
print(df_newsapi.head())
df_newsapi.to_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_newsapi_final.csv', index=False)


print('gdelt')
df_gdelt = pd.read_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_gdelt_output.csv', header=0)
df_gdelt = df_gdelt.drop(['url', 'url_mobile', 'socialimage', 'language', 'domain_key'], axis=1)
new_column_names = {'title': 'Title',
                    'seendate': 'Date',
                    'sourcecountry': 'Country',
                    'text': 'Text',
                    'domain': 'Source'}
df_gdelt = df_gdelt.rename(columns=new_column_names)
df_gdelt = df_gdelt.dropna(subset=['Text'])

df_gdelt['Title'] = df_gdelt['Title'].astype(str)
df_gdelt['Text'] = df_gdelt['Text'].astype(str)

## Labeling data
# Parts of speech tagging
print('gdelt additional labels')
df_gdelt['POS_tag'] = [return_pos_tag(text) for text in df_gdelt['Text']]

# Sentiment
df_gdelt['Title_nltk'] = get_nltk_sentiment(df_gdelt['Title'])
df_gdelt['Title_vader'] = get_vader_sentiment(df_gdelt['Title'])
df_gdelt['Text_nltk'] = get_nltk_sentiment(df_gdelt['Text'])
df_gdelt['Text_vader'] = get_vader_sentiment(df_gdelt['Text'])

df_gdelt.to_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_gdelt_final.csv', index=False)

df_gdelt = pd.read_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_gdelt_final.csv', header=0)
df_gdelt = df_gdelt[df_gdelt['Text'].apply(is_english)]
df_gdelt['Date'] = [datetime.strptime(date, '%Y%m%dT%H%M%SZ').date() for date in df_gdelt['Date']]

print(df_gdelt)
df_gdelt.to_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_gdelt_final.csv', index=False)


df_newsapi = pd.read_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_newsapi_final.csv', header=0)
df_gdelt = pd.read_csv('C:\\Users\\Jeffry J. Bai\\Desktop\\project_gdelt_final.csv', header=0)

df_newsapi['Date'] = pd.to_datetime(df_newsapi['Date'])
df_gdelt['Date'] = pd.to_datetime(df_gdelt['Date'])


print('Generating Wordcloud')
## Generate wordclouds
wordcloud = get_wordcloud(df_newsapi['Text'])

for year in range(2017, 2024):
    print(year)
    filtered_df = df_gdelt[df_gdelt['Date'].dt.year == year]
    print(filtered_df)
    wordcloud = get_wordcloud(filtered_df['Text'])


print('Summary Statistics')

df_newsapi = df_newsapi[df_newsapi['AllSides Bias Rating'] != 'Center']

mapping = {
    "Lean Left": "Left",
    "Lean Right": "Right"}

df_newsapi["AllSides Bias Rating"] = df_newsapi["AllSides Bias Rating"].replace(mapping)

summary_statistics(df_newsapi,
                   'Title_nltk',
                   'Summary_nltk',
                   'Text_nltk',
                   'Title_vader',
                   'Summary_vader',
                   'Text_vader',
                   title=f"Right Bias Summary Statisics")


for year in range(2017, 2024):
    print(year)
    filtered_df = df_gdelt[df_gdelt['Date'].dt.year == year]
    summary_statistics(filtered_df,
                       'Title_nltk',
                       'Text_nltk',
                       'Title_vader',
                       'Text_vader',
                       title=f"{year} Summary Statistics")


    plot_summary_trend(filtered_df,
                        "Date",
                       'Title_nltk',
                       'Text_nltk',
                       'Title_vader',
                       'Text_vader')


print('Vectorization')
## Vectorization
df_newsapi["Text"] = [text if isinstance(text, str) else '' for text in df_newsapi["Text"].tolist()]
df_newsapi = df_newsapi[df_newsapi['Text'] != '']

print(df_newsapi.describe())


#
#
# CountVectorizer - Predict Bias
#
#
dataset = "newsapi"



for vectorizer, v_name in zip([cv, tfidf], ["cv", "tfidf"]):
    fit = vectorizer.fit_transform(df_newsapi["Text"].tolist())
    feature_names = vectorizer.get_feature_names_out()
    df_vector = pd.DataFrame(fit.toarray(), columns=feature_names)


    lda_model.fit(df_vector)
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"{v_name} Topic #{topic_idx + 1}:")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-20:-1]]))
        print()


    y = df_newsapi["AllSides Bias Rating"].tolist()

    print(f'{dataset} {v_name} with feature length of {len(feature_names)}')
    x_train, x_test, y_train, y_test = train_test_split(fit, y, test_size=0.05, random_state=736)

    mnb.fit(x_train, y_train)
    y_pred = mnb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(f'\tAccuracy MNB {v_name}: {accuracy}')
    print(cm, "\n")
    plot_confusion_matrix(cm, set(y), title=f"MNB {dataset} {n_gram} {v_name}")


    # Get the learned coefficients from MNB
    top_words_0 = [feature_names[i] for i in mnb.feature_log_prob_[0].argsort()[-10:]]
    top_words_1 = [feature_names[i] for i in mnb.feature_log_prob_[1].argsort()[-10:]]
    class_indices = mnb.classes_
    print(f'\tMost indicative words for {class_indices[0]} bias:', top_words_0)
    print(f'\tMost indicative words for {class_indices[1]} bias:', top_words_1, '\n')


    # DTC
    print(f"DTC {v_name}")
    dtc.fit(x_train, y_train)

    fig, ax = plt.subplots(figsize=(12, 12))
    tree.plot_tree(dtc, feature_names=feature_names, class_names=y, filled=True, ax=ax)

    # Display the plot
    plt.show()

    y_pred = dtc.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print('\tAccuracy:', accuracy)
    print(cm, "\n")
    plot_confusion_matrix(cm, set(y), title=f"DTC {dataset} {n_gram} {v_name}")

    feature_importances = dtc.feature_importances_
    important_features = [feature_names[i] for i in feature_importances.argsort()[-10:]]
    print('\tMost important features:', important_features, '\n')


    for kernel in ["linear", "poly", "rbf"]:
        print(kernel)
        svm = SVC(kernel=kernel)
        svm.fit(x_train, y_train)

        print("predict")
        y_pred = svm.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)

        print(f'\tAccuracy svm {kernel}:', accuracy)
        print(cm, "\n")
        plot_confusion_matrix(cm, set(y), title=f"{dataset} {n_gram} {v_name} {kernel} CM")


#
#
# CountVectorizer - Predict Year
#
#


print("\n\n\n")
df_gdelt["Text"] = [text if isinstance(text, str) else '' for text in df_gdelt["Text"].tolist()]
df_gdelt = df_gdelt[df_gdelt['Text'] != '']
df_gdelt['Year'] = pd.to_datetime(df_gdelt['Date']).dt.year

summary_statistics_trend(df_gdelt,
                         'Year',
                         'Title_nltk',
                         'Text_nltk',
                         'Title_vader',
                         'Text_vader',
                         title=f"2017 to 2023 Trend Summary Statistics")


for year in range(2017, 2024):
    print(f"\n\n\n------------------------------{year}")
    df_gdelt_y = df_gdelt[df_gdelt['Year'].isin([year])]
    dataset = f" 2017 - {year} gdelt"

    get_wordcloud(df_gdelt_y["Text"])

    for vectorizer, v_name in zip([cv, tfidf], ["cv", "tfidf"]):
        print(f"\n---{v_name}")
        fit = vectorizer.fit_transform(df_gdelt_y["Text"].tolist())
        feature_names = vectorizer.get_feature_names_out()
        df_vector = pd.DataFrame(fit.toarray(), columns=feature_names)


        y = df_gdelt_y['Year']

        print(f'{dataset} {v_name} with feature length of {len(feature_names)}')
        x_train, x_test, y_train, y_test = train_test_split(fit, y, test_size=0.05, random_state=736)

        mnb.fit(x_train, y_train)
        y_pred = mnb.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)

        print(f'\tAccuracy MNB {v_name}: {accuracy}')
        print(cm, "\n")
        plot_confusion_matrix(cm, set(y), title=f"MNB {dataset} {n_gram} {v_name}")


        # Get the learned coefficients from MNB
        top_words_0 = [feature_names[i] for i in mnb.feature_log_prob_[0].argsort()[-10:]]
        top_words_1 = [feature_names[i] for i in mnb.feature_log_prob_[1].argsort()[-10:]]
        class_indices = mnb.classes_
        print(f'\tMost indicative words for {class_indices[0]} bias:', top_words_0)
        print(f'\tMost indicative words for {class_indices[1]} bias:', top_words_1, '\n')

        # DTC
        print(f"DTC {v_name}")
        dtc.fit(x_train, y_train)

        y_pred = dtc.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print('\tAccuracy:', accuracy)
        print(cm, "\n")
        plot_confusion_matrix(cm, set(y), title=f"DTC {dataset} {n_gram} {v_name}")


        for kernel in ["linear", "poly", "rbf"]:
            print(kernel)
            svm = SVC(kernel=kernel)
            svm.fit(x_train, y_train)

            print("predict")
            y_pred = svm.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)

            print(f'\tAccuracy svm {kernel}:', accuracy)
            print(cm, "\n")
            plot_confusion_matrix(cm, set(y), title=f"{kernel} {dataset} {n_gram} {v_name} CM")

