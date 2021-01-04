# We'll start by reading in the corpus, which preserves word order
import pandas as pd
from textblob import TextBlob


def analyze_sentiment(df):
    df.to_pickle("reuters_data.pkl")
    data = pd.read_pickle("reuters_data.pkl")
    print(data)

    # Create quick lambda functions to find the polarity and subjectivity of each routine
    # Terminal / Anaconda Navigator: conda install -c conda-forge textblob

    pol = lambda x: TextBlob(x).sentiment.polarity
    data['polarity'] = data['processed_article'].apply(pol)

    # Let's plot the results
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = [10, 8]

    for index, article in enumerate(data.index):
        x = data.polarity.loc[article]
    data.to_csv('Polarity.csv', index=False)

