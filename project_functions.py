import datetime


def gdelt_query(query,
                start_date = datetime.date.today(),
                end_date = datetime.date.today(),
                domain = [],
                country = []):
    from gdeltdoc import GdeltDoc, Filters

    filter = Filters(
        keyword = query,
        start_date = start_date,
        end_date = end_date,
        domain = domain,
        country = country
    )

    gd = GdeltDoc()

    # Search for articles matching the filters
    articles = gd.article_search(filter)

    print(f"{len(articles.index)} added")

    return articles

# Function scraps elements from of table of a given URL and outputs the table as df.


def scrape_table_elements(url, table_id=None, table_class=None):
    import requests
    import pandas as pd

    from bs4 import BeautifulSoup

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    data = []
    header = []

    if table_id and table_class:
        tables = soup.find_all('table', id=table_id, class_=table_class)
    elif table_id:
        tables = soup.find_all('table', id=table_id)
    elif table_class:
        tables = soup.find_all('table', class_=table_class)
    else:
        tables = soup.find_all('table')

    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            headers = row.find_all('th')
            if headers:
                header = [header.get_text(strip=True) for header in headers]

            cells = row.find_all('td')
            if cells:
                row_data = []
                for cell in cells:
                    if cell.get_text(strip=True):
                        row_data.append(cell.get_text(strip=True))
                    elif cell.find('img'):
                        alt_text = cell.find('img')['alt']
                        row_data.append(alt_text)
                data.append(row_data)

    df = pd.DataFrame(data, columns=header)

    return df


def return_pos_tag(text):
    import nltk

    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)


def cosine_similarity_score(text1, text2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    # Print the similarity score
    return cosine_sim[0][0]


# Function generates a dataframe from NewsAPI based on a given query
def get_news_dataframe(query,
                       key = 'Enter your key',
                       domains=''):
    import pandas as pd

    from newsapi import NewsApiClient

    newsapi = NewsApiClient(api_key = key)

    column_names = ['Title', 'Source', 'Author', 'Summary', 'Date', 'Text']

    df = pd.DataFrame(columns = column_names)

    response = newsapi.get_everything(q = query,
                                      domains=domains,
                                      language = 'en')

    for article in response['articles']:
        url = article['url']
        try:
            text = get_article_text(url)

            title = article['title']
            author = article['author']
            source = article['source']['name']
            summary = article['description']
            date = article['publishedAt']

            row_data = pd.Series([title, source, author, summary, date, text],
                                 index = column_names)
            df = df._append(row_data,
                            ignore_index=True)
            print(row_data)
        except:
            print('Skip')

    return df


def get_top_news_dataframe(query,
                       key = 'Enter your key',
                       country='us'):
    import pandas as pd

    from newsapi import NewsApiClient

    newsapi = NewsApiClient(api_key = key)

    column_names = ['Title', 'Source', 'Author', 'Summary', 'Date', 'Text']

    df = pd.DataFrame(columns = column_names)

    response = newsapi.get_top_headlines(q = query,
                                         country=country,
                                         language = 'en')

    for article in response['articles']:
        url = article['url']
        try:
            text = get_article_text(url)

            title = article['title']
            author = article['author']
            source = article['source']['name']
            summary = article['description']
            date = article['publishedAt']

            row_data = pd.Series([title, source, author, summary, date, text],
                                 index = column_names)
            df = df._append(row_data,
                            ignore_index=True)
        except:
            print('Skip')

    return df


def get_article_text(url):
    from newspaper import Article

    article = Article(url)
    article.download()
    article.parse()

    return article.text


def get_nltk_sentiment(text_list):
    from textblob import TextBlob
    import numpy as np

    result = [TextBlob(text).sentiment.polarity for text in text_list]

    return result


def get_vader_sentiment(text_list):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import numpy as np

    analyzer = SentimentIntensityAnalyzer()
    result = [analyzer.polarity_scores(text)['compound'] for text in text_list]

    return result


def stemmer_tokenizer(text):
    import re
    from nltk.stem import PorterStemmer

    porter = PorterStemmer()

    tokens = re.findall(r'\b(?![0-9]+\b)[A-Za-z]+\b', text.lower())

    final_tokens = [porter.stem(token) for token in tokens]

    return final_tokens


def is_english(text):
    from langdetect import detect

    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False


def get_wordcloud(list):
    import matplotlib.pyplot as plt

    from wordcloud import WordCloud, ImageColorGenerator

    corpus = ""

    for text in list:
        corpus += f" {text}"

    wordcloud = WordCloud(width=800, height=800, max_words=100, background_color="white").generate(corpus)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    return wordcloud


def summary_statistics(df, *columns, title=None):
    import plotly.graph_objects as go

    data = [df[col].astype(float).dropna().tolist() for col in columns]

    fig = go.Figure()

    for col, dataset in zip(columns, data):
        fig.add_trace(go.Box(y=dataset, name=col))

    fig.update_layout(title_text=title)

    fig.show()


def summary_statistics_trend(df, x_column, *y_columns, title=None):
    import plotly.graph_objects as go

    fig = go.Figure()

    for y_col in y_columns:
        fig.add_trace(go.Box(y=df[y_col], x=df[x_column], name=y_col))


    fig.update_layout(title_text=title, xaxis_title='Date', yaxis_title='Value')

    fig.show()

def plot_summary_trend(df, date, *columns):
    import plotly.graph_objects as go
    import pandas as pd

    df['Year'] = pd.to_datetime(df[date]).dt.year

    fig = go.Figure()
    shapes = ['circle', 'square', 'triangle-down', 'triangle-up']

    for i, col in enumerate(columns):
        fig.add_trace(go.Scatter(
            x=df[date],
            y=df[col],
            mode='markers',
            name=col,
            marker=dict(symbol=shapes[i % len(shapes)], size=2, opacity=0.7)
        ))

        # Calculate trendline values by year
        trendline = df.groupby('Year')[col].mean().reset_index()

        fig.add_trace(go.Scatter(
            x=trendline['Year'],
            y=trendline[col],
            mode='lines',
            name=f'Trendline {col}'
        ))

    fig.update_layout(title='Summary Statistics Trend', xaxis_title='Date', yaxis_title='Value')
    fig.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(f"C:\\Users\\Jeffry J. Bai\\Documents\\IST736\\{title}")

