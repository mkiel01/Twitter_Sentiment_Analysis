from preprocessing import *
from tokenization import *
from stopwords import *
from network import *

data = load_data(path = '/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/training.1600000.processed.noemoticon.csv')
show_head(data, message= "of original data")
data = columns_renaming(data)

data = change_size(0.01, original_data = data)

show_head(data, message= "of original data with changed size")
tokenized_data = tokenized(data)
save_data_to_csv(tokenized_data, path = "data/tokenized_twitter.csv")
show_head(tokenized_data, message = "of tokenized twitter")
no_stopwords_twitter = tokenized_data.copy() 
no_stopwords_twitter["tweet"] = tokenized_data['tweet'].apply(remove_stopwords)
show_head(no_stopwords_twitter, message = "of stopworks twitter")
save_data_to_csv(no_stopwords_twitter, path="/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/no_stopwords_twitter.csv")
final_senstiment = run_model(data)
final = combine(data, final_senstiment)
show_head(final, message = "of final data")
calc_sentiment(data)
save_data_to_csv(final, path="/Users/michalkielkowski/Desktop/infa-all/magisterka/eksploracja_danych/Twitter_Sentiment_Analysis/data/twitter_with_calc_sensti.csv")
