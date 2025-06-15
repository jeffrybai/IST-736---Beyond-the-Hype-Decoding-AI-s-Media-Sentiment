**Bai** , Jeffry; **Burke** , Dan; **Callahan** , Claire; **Castellanos** , Kendra
jbai06@syr.edu, dburke01@syr.edu, ccall100@syr.edu, kecastel@syr.edu
Final Project
Artificial Intelligence
**MNB, DTC, SVM**
_June 17th, 2023
Presentation Link_
**BEYOND THE HYPE: DECODING AI'S MEDIA SENTIMENT
INTRODUCTION**
The start of Artificial intelligence (AI) dates back to the mid 1900s when the
concept of creating machines capable of intelligent behavior first emerged. Alan
Turing was a visionary thinker and pioneer in the field of artificial intelligence; he
left an incredible mark on the development of modern technology that continues
to shape today’s world. Turing's most notable contribution was the proposal of the
"Turing Test," a thought experiment that assessed a machine's ability to exhibit
human-like behavior. Additionally, his concept of the universal computing
machine, known as the Turing machine, demonstrated the potential for a single
machine to perform a multitude of tasks. Turing also explored the idea of
machine learning, envisioning a future where machines could learn from and
adapt to new information. These foundational concepts laid the groundwork for
AI research and remain vital to the understanding of intelligent systems today.
Today, Machine Learning is no longer just an ideation but a reality that has
revolutionized the field of artificial intelligence since Alan Turing's passing.
Significant advancements have been made, driven by the exponential growth of
computing power and the availability of vast amounts of data. Machine Learning
techniques, such as deep neural networks, have achieved remarkable
breakthroughs in various domains, including computer vision, natural language
processing, and advancements in medicine. These accomplishments have led to
practical applications such as self-driving cars, voice assistants, personalized
recommendations, and medical diagnoses.
The advent of the World Wide Web has played a pivotal role in accelerating the
development of AI, ushering in a new era of connectedness and accessibility. The
web has facilitated collaboration and knowledge sharing among researchers and
practitioners globally, fostering a vibrant community that has propelled AI
forward. For example, a group of researchers was able to produce Alpaca 7B for
under $600 with its comparable performance to Open AI’s Generative
Pre-Trained (GPT) Model 3.5. Online platforms and open-source initiatives have
democratized access to AI tools, frameworks, and libraries, empowering
developers and entrepreneurs to leverage AI technology and drive innovation
across various sectors.
Initially, there was tremendous optimism and excitement surrounding AI. Many
believed that it would revolutionize various industries, improve efficiency, and
solve complex problems. However, as AI progressed, concerns and debates


emerged. People began questioning the ethical implications and potential risks
associated with the development of highly intelligent machines. Science fiction
and popular culture played a role in shaping public perception, often portraying
AI as a potential threat to humanity. Additionally, issues such as privacy
infringement, algorithmic bias, and the potential for unintended consequences can
arise. The increasing automation of jobs raises concerns about unemployment and
socioeconomic inequalities. There are also fears around the lack of transparency
and interpretability of complex AI systems, which can make it challenging to
understand and rectify potential errors or biases. Lastly, is the application of AI
weapon systems; such that the development of the tool is about hurting rather
than helping people. Addressing these downsides requires a balanced and
thoughtful approach, involving collaboration between policymakers, researchers,
and industry leaders to ensure the responsible development and deployment of AI
technologies that prioritize human well-being and societal benefit.
Media analysis of the topic of AI can be immensely helpful in understanding the
impact, advancements, and implications of artificial intelligence. Overall, people's
feelings about AI are diverse and multifaceted. While there is enthusiasm for its
transformative potential, there is also wariness about the ethical and societal
implications. Striking a balance between harnessing the benefits of AI and
addressing the associated challenges remains a crucial task for researchers,
policymakers, and society as a whole. The ongoing dialogue and collaboration
among various stakeholders will shape the future trajectory of AI, ensuring that it
aligns with human values and benefits humanity as a whole By examining media
coverage on AI, one can observe emerging trends, breakthroughs, and the
evolving discourse surrounding AI technology. It helps policymakers, businesses,
and researchers stay informed about the latest developments and make data driven
decisions. Furthermore, media analysis can shed light on the responsible
deployment and governance of AI, fostering transparency, accountability, and
public trust.
**ANALYSIS
DATA PREPARATION**
Data Collection
For the purposes of this analysis, data points such as news title, article text, source
& url, publish date, and media bias are required. In order to get media bias, a
custom table scrapping function was written to gather data from AllSides.com.
The media dataframe contains columns [ _News Source, Bias Rating, Domain_ ].


_Fig 1. Sample dataframe scraped from AllSides Media Bias_
To obtain news data, two different libraries were used to collect the news data on
the topic of Artificial Intelligence - **newsapi** and **gdeltdoc**. The domain obtained
from media bias is then given to the aforementioned APIs as a parameter. This
resulted in two different data frames. The first dataframe generated from **newsapi**
contains columns [ _Title, Source, Author, Summary, Date, URL, Domain, Country,
News Source_ ]. To obtain the text of the articles, the **Article** function from
**newspaper** library was leveraged to remove html elements and obtain text of the
article. The **newsapi** and **bias** data frames are then merged together on Domain
as a key to label each article as either [ _Left, Lean Left, Center, Lean Right, Right_ ].
The end result is dataframe for bias labeled data for analysis. This process is
repeated for **gdeltdoc** to obtain historical data.
_Fig 2. Intermediate dataframe of data from_ **_newsapi_** _&_ **_AllSides.com_** _Media Bias_
Data Cleaning
Since **newsapi** and **gdeltdoc** are intermediate dataframes, some urls are not
available at the time of request. The dataframes are iterated through to remove
blank values or none english texts. In order to generate better predicts, “Lean”
was removed from bias label to reduce the label choices. Date value was also
formatted from '%Y-%m-%dT%H:%M:%SZ' formate to standard date string to
facilitate processing. When initializing **CountVectorizer** (CV) and **Term
Frequency - Inverse Document Frequency** (TFIDF), stopwords and numbers
were removed. The tokens were also stemmed to reduce vocabulary and improve
models’ efficiency.


## DATA EXPLORATION

The **newsapi** and **gdeltdoc** data frames are 586 and 12,728 documents
respectively.
_Fig 3. Wordcloud generated with data from_ **_newsapi_**
_Fig 4. Wordcloud generated with data from 2017 to 2023_ **_gdelt_**
**MODELS & METHODS**
To gather statistics on text sentiment, 2 different models were applied to the title,
summary, and text of the articles. **TextBlob** is a popular library used for natural
language processing tasks, including sentiment analysis. It provides a simple and
intuitive way to determine the sentiment polarity ( _positive, negative, or neutral_ )


of a given text or document. **VADER** (Valence Aware Dictionary and sEntiment
Reasoner) is a rule-based sentiment analysis tool specifically designed for social
media texts. It can accurately analyze the sentiment of text by considering both
the polarity (positive or negative) and intensity (strength) of the sentiment
expressed. To compare model performance, the normalized indexes of
**TextBlob’s sentiment polarity** were compared to **VADER’s compound polarity
scores.
CountVectorizer (CV)** is a commonly used technique in natural language
processing that converts a collection of text documents into a matrix representing
the frequency of occurrence of each word or term within the documents, enabling
further analysis and modeling based on the document-term matrix. **Term
Frequency-Inverse Document Frequency (TF-IDF)** is a numerical statistic
used to evaluate the importance of a term within a document or a collection of
documents. To get a full picture for the analysis, both CV and TFIDF were
initialized with unigram, bigram, trigram, and quadrigram.
For bias prediction both CV and TFIDF were trained with Multinomial Naive
Bayes, Decision Tree Classifier, and Support Vector Machine. **Multinomial
Naive Bayes (MNB)** is a probabilistic classifier commonly used for text
classification tasks, particularly when dealing with features that represent discrete
word counts or frequencies. **Decision Tree Classifier (DTC)** is a predictive
modeling algorithm that uses a tree-like structure to make decisions by splitting
data based on features and their thresholds, enabling both classification and
regression tasks. **Support Vector Machine (SVM)** is a powerful supervised
learning algorithm that can be used for both classification and regression tasks,
aiming to find an optimal hyperplane that maximally separates different classes in
the data by mapping it into a higher-dimensional feature space.
Finally, **Latent Dirichlet Allocation (LDA)** is used for topic modeling. **LDA** is
a generative probabilistic model used for topic modeling, capable of uncovering
latent topics within a collection of documents by assigning probabilities to words
and documents based on statistical patterns and co-occurrences.
**ANALYSIS GOALS & PARAMETERS**
The goal is to perform a wide range of analysis of public sentiment as expressed
through news outlet on the subject of Artificial Intelligence. In addition to
sentiment trend, it would be interesting to see if there are variances in sentiment
through the lens of political views. Another area of interest is to compare model
performance against one another to see which model performance for predictions
under what circumstances. SVMs are initialized and trained on default
parameters such that it is easier to compare them against one another.


## RESULTS

## SENTIMENT ANALYSIS

```
Fig 5. Summary statistics of TextBlob v. VADER sentiment scores
To gain a holistic comprehension of the sentiment scoring methods, see
comparison of summary statistics for TextBlob and VADER in Fig 3. Generally
speaking, media sentiment on the subject of A.I. trend positive. One observation
is that title and summary appear to trend more negatively when compared to text
of the article; this observation can be made for both methods. For TextBlob, the
spread of the sentiment distribution is smaller as the length of the corpus
increases from title to text. The exact opposite trend is true for VADER, as the
length of the corpus increases, the spread of the distribution also increases. One
explanation for this observation could be a result of how these sentiment methods
are engineered. VADER is designed for short informal text; when analyzing
formal long form documents, the models produce results that are more polarized.
The polarizing effect can also be observed when examining sentiment distribution
over time in Fig 4. When looking at TextBlob, one can observe that the
distribution is around the center, indicative of journalistic neutrality, with the title
experiencing more outliers.
Fig 6. Scatter plot of TextBlob v. VADER sentiment scores over time.
```

## BIAS ANALYSIS

Considering the difference in TextBlob and VADER, it may be prudent to draw
conclusions about title sentiment from VADER and text sentiment from TextBlob.
Another interpretation of VADER sentiment can also be that the result is more
polarized. Armed with the knowledge, the comparison of summary statistics
across political bias can be seen in _Fig. 4 & 5_. From TextBlob, Left bias media
have greater spread of title sentiment, but majority are sentimentality neutral.
Whereas Right bias media have a smaller overall spread, but demonstrate a
greater degree of variance and standard deviation. This observation is also
supported by the VADER sentiment score. From VADER, titles from Left bias
data also trend more negative compared to Right bias. With the text of the
articles, TextBlob sentiment exhibited greater spread on Left bias. The opposite
is true when looking at VADER compound scores. It is difficult to make any
conclusive statements in regards to the text sentiment in relation to political bias.
Both biases average positive sentiment across the political spectrum.
_Fig 7. Comparison of TextBlob summary statistics between Left v Right Bias
Fig 8. Comparison of VADER summary statistics between Left v Right Bias_


_Fig 9. Comparison of average between Left v Right Bias for both TextBlob and VADER_
**TREND ANALYSIS**
_Fig 10. Summary statistics trend for TextBlob text sentiment._


_Fig 11. Summary statistics trend for TextBlob & VADER text sentiment._
There are no observable trend over the 5 year span. One feature of interest of
note is that between 2018 and 2020, both models exhibits greater spread. This
time period coincides with the COVID-19; while it is not enough to draw any
conclusions, the feature is interesting to note.
**BIAS PREDICTION**
The best performing model in the train split random state 736 for bias prediction
was DecisionTreeClassifier with 4-gram with an accuracy of 93.33%. In fact,
DTC consistently outperformed MNB in n-gram of up to 4 for this particular
random state. MNB for CV’s performance fell as n-gram increased. This was the
opposite trend for MNB TFIDF.
_Fig 12. Confusion Matrix for DTC 4-gram TFIDF bias prediction._


In fact, DTC performed the best in any n-gram vectorizer. Verifying this find
with a 10-fold cross validation score, DTC on average scored better than any
other model, with diminishing returns at _n_ -gram of greater than 4. As for MNB,
it performed best with CV 1-gram with a falling accuracy as _n_ increases. MNB
observes an increase in performance for TFIDF at 2-gram & 3-gram, but reduced
accuracy after _n_ = 3. Out of the SVMs, Linear model consistently outperformed
polynomial (poly) and Radial Basis Function (RBF) with default parameters.
Poly performed the worst out of the SVMs; however, since these SVM were
initialized with default parameters, this is only indicative that for poly degree of 3
is not suitable for this prediction. Based on the findings in _Fig. 13 & 14_ , it is fair
to conclude that there are few advantages to vectorize n-grams where n > 3.
_Fig 13. CV Model 10-fold cross validation performance.
Fig 14. TFIDF Model 10-fold cross validation performance._


## BIAS INDICES

When generating key features for the bias indices, an interesting observation
emerged when considering n-grams of size two or greater. The inclusion of media
sources and journalist names as tokens proved to be highly indicative of political
bias. Additionally, when examining n-grams of size three or greater, certain terms
such as "Large Language Model" and "Generative A.I." appeared as indicators of
left bias, while "Natural Language Processing" emerged as a marker of right bias.
While these findings offer potential insights, it is crucial to acknowledge potential
data limitations before drawing any conclusions.
While both political spectrums cover the risks involved with A.I., it is more
indicative of Left bias if the article discusses the regulation of A.I. This is in
contrast to Right bias where they approach the subject by discussing people’s fear
of the technology. This is in line with the understanding of the political paradigm
for these biases. Right wing politics is about free-market in contrast to Left wing
politics that advocates for greater governmental oversight.
_Fig 15. Table of example indices for political bias_


## LATENT DIRICHLET ALLOCATION LATENT TOPICS

```
Fig 16. LDA 1 - AI application in the work force; writers’ strike
Fig 17. LDA 2 - Geopolitical application of AI to supremacy
```

_Fig 18. LDA 3 - AI application in Medicine
Fig 19. LDA 4 - AI application from technology and business companies_


_Fig 20. LDA 5 - AI application in war
Fig 21. LDA 1 - Risks of AI application_


_Fig 22. LDA 7 - AI and law; regulations, court cases; AI application in the legal industry_
**CONCLUSIONS**
In this analysis of media sentiment on the topic of Artificial Intelligence (AI), it
was found that both left-wing and right-wing media generally expressed positive
sentiment. However, there were interesting observations when considering the
sentiment of article titles compared to the sentiment of the article text. The
sentiment scores of titles tended to be lower than those of the corresponding text,
indicating potential variations in the framing or tone used in headlines. This also
goes to show that media outlets often use attention grabbing titles that can be
controversial. This pulls readers in, even if the rest of the article is not as
politically aligned.
Moreover, the analysis did not reveal any obvious differences in the average
sentiment between left-wing and right-wing media. This suggests that when it
comes to discussions on AI, both political spectrums exhibit similar overall
sentiment - at least within the analyzed dataset. There is also no obvious feature
when it comes to sentiment trends since 2017. It is important to note that media
sentiment can vary widely based on specific events, contexts, and the diversity of
media sources considered. And as such, additional research would need to be
conducted on the topic to compare additional ideations in sentiment..
Furthermore, exploring key features lead to the discovery of latent indices within
the coverage of AI. Notably, topics related to government regulation emerged as
an indicator of left bias. This finding aligns with the understanding of left-wing
politics advocating for greater governmental oversight and intervention.
Conversely, right-wing media showed a focus on public fears and concerns about
AI, which is in line with the political inclination towards free-market principles
and limited government intervention. These slight differences in concepts
generally reflect what researchers would expect to see in oppositional media
outlets.
Notably, it is crucial to recognize the complexity and diversity within both
left-wing and right-wing media landscapes. While these findings shed light on
sentiment and topic preferences within the analyzed dataset, it is important to
consider that media sentiment and bias are multifaceted, influenced by a range of
factors, and can vary across different media outlets and individual journalists. It is
also critical to recognize that media representations of AI, on both sides, are often
influenced by sensationalism, entertainment value, and the need to attract
viewers. As AI continues to develop, it is crucial for the media to present a
balanced and informed perspective that educates the public about the capabilities,
limitations, and ethical considerations surrounding AI technology.
Overall, this analysis provides valuable insights into media sentiment on AI and
its relation to political bias. It highlights the similarities in average sentiment
between left-wing and right-wing media, while also showcasing the influence of
political paradigms through the identification of n-gram features related to
government regulation and public fears. Further research and analysis,
incorporating larger and more diverse datasets, would contribute to a deeper
understanding of media sentiment and its connection to political biases in the
context of AI coverage. However, it is clear that AI has distinguished itself as
being beyond the hype and is here to stay, at least in all media outlets, regardless
of where they lean.
