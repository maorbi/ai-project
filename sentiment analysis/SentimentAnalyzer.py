from collections import Counter
from string import punctuation
import numpy as np

import inline as inline
import matplotlib as matplotlib
import pandas as pd
import matplotlib.pyplot as plt




def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            new = zeroes + review
        elif review_len > seq_length:
            new = review[0:seq_length]
        features[i, :] = np.array(new)

    return features


def main():
    with open('headers.csv', 'r',encoding='utf-8-sig') as f:
        news = f.read()
    news = news.lower()
    plain_news = ''.join([c for c in news if c not in punctuation])

    with open('reuters_positive.csv', 'r', errors="ignore") as f:
        positive_vals=f.read()
    with open('reuters_negative.csv', 'r', errors="ignore") as f:
        negative_vals=f.read()
    #print(plain_news)
    news_split = plain_news.split('\n')
    i = 0
    for article in news_split:
        news_split[i] = article + " "
        i = i + 1
    news_split2 = ''.join(news_split)
    words = news_split2.split()
    count_words = Counter(words)
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    #now we shall create the reviews vector
    headers_vector = []
    for news in news_split:
        rev = [vocab_to_int[w] for w in news.split()]
        headers_vector.append(rev)
    print(headers_vector[0:4])
    headers_len = [len(x) for x in headers_vector]
    pd.Series(headers_len).hist()
    plt.show()
    pd.Series(headers_len).describe()
    features = pad_features(headers_vector, 25)
 #   print(features)
    print(features[:10, :])

if __name__ == "__main__":
    main()