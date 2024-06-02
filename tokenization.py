import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


tk = TweetTokenizer()

def tokenized(data):


    tokenized_twitter = data.copy()

    tokenized_twitter["tweet"] = data["tweet"].apply(tk.tokenize)
    return tokenized_twitter


