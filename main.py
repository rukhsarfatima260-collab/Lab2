
import re, string, random
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download('twitter_samples')
nltk.download('stopwords')

# ── process_tweet ──────────────────────────────────────────
def process_tweet(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    sw = stopwords.words('english')
    tokens = [w for w in tokens if w not in sw and w not in string.punctuation]
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in tokens]

# ── build_freqs ────────────────────────────────────────────
def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

# ── Load data ──────────────────────────────────────────────
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
tweets = all_positive_tweets + all_negative_tweets
labels = np.append(np.ones(len(all_positive_tweets)), np.zeros(len(all_negative_tweets)))

# ── Build frequency dict ───────────────────────────────────
freqs = build_freqs(tweets, labels)
print(f'type(freqs) = {type(freqs)}')
print(f'len(freqs) = {len(freqs)}')

# ── Word count table ───────────────────────────────────────
keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']

data = []
for word in keys:
    pos = freqs.get((word, 1), 0)
    neg = freqs.get((word, 0), 0)
    data.append([word, pos, neg])

# ── Scatter plot (log scale) ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
x = np.log([d[1] + 1 for d in data])
y = np.log([d[2] + 1 for d in data])
ax.scatter(x, y)
plt.xlabel("Log Positive count")
plt.ylabel("Log Negative count")
for i in range(len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)
ax.plot([0, 9], [0, 9], color='red')
plt.show()