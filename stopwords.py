import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stop_words_cached = stopwords.words('english')
tk = TweetTokenizer()

def remove_stopwords(text_array):
    fresh_text_array = []
    for word in text_array:
      lowercase_word = word.lower()
      if lowercase_word not in stop_words_cached:
        fresh_text_array.append(lowercase_word)
    return fresh_text_array
