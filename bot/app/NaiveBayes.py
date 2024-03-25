
import pandas as pd
class NaiveBayesClassifier:
    def __init__(self):
        self.vocab = set()  # Vocabulary
        self.negative_word_count = 0 # Total number of words for each class
        self.positive_word_count = 0

        self.positive_words = {}  # Word frequencies for each class
        self.negative_words = {}

        self.positive_bigram_count = 0
        self.negative_bigram_count = 0
        self.positive_bigrams = {}
        self.negative_bigrams = {}

    def fit(positive_text, negative_text):
        positive_text = positive_text



df = pd.read_csv("../tweets-cleaned.csv", encoding='ISO-8859-1')
df = df.iloc[:, [0, 2]]
df['target'].replace(4, 1, inplace=True) # Replace all the 4s with 1s

text, sentiment = list(df['clean']), list(df['sentiment']) # Turn these both into lists
text = [str(item) for item in text]
positive_text = text[800000:] # Dataset arranged in a way such that the first half is negative and second half is positive
negative_text = text[:800000]
print(positive_text)

