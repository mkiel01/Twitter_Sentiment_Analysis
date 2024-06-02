from preprocessing import *
from tokenization import *
from stopwords import *
from network import *

data = load_data(path = '/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/training.1600000.processed.noemoticon.csv')
show_head(data, message= "of original data")
data = columns_renaming(data)

data = change_size(0.1, original_data = data)

show_head(data, message= "of original data with changed size")
tokenized_data = tokenized(data)
save_data_to_csv(tokenized_data, path = "data/tokenized_twitter.csv")
show_head(tokenized_data, message = "of tokenized twitter")
no_stopwords_twitter = tokenized_data.copy() 
no_stopwords_twitter["tweet"] = tokenized_data['tweet'].apply(remove_stopwords)
show_head(no_stopwords_twitter, message = "of stopworks twitter")
save_data_to_csv(no_stopwords_twitter, path="/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/no_stopwords_twitter.csv")
twitter = assign_new_colum(no_stopwords_twitter, name="outer_sentiment")
show_head(twitter, message = "of final data")
save_data_to_csv(twitter, path="/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/twitter_final.csv")
run_model(data)
save_data_to_csv(twitter, path="/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/twitter_final1.csv")
show_head(data, message = "of final data")
data = "/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/twitter_final1.csv"
calc_sentiment(data)
