import re # Regex for the Naive Bayes model
from nltk.stem import WordNetLemmatizer # WordNetLemmatizer for the Naive Bayes model
from nltk.tokenize import word_tokenize # Also for Naive Bayes model
class NaiveBayesClassifier:
    def __init__(self):
        with open("../stop-words.txt") as f:
            self.stopwords = [line.strip() for line in f]
        self.stopwords = set(self.stopwords) # Set of stop words to check against for the clean method for each prediction, keep in mind this model takes in cleaned data so should only be used for prediction

        with open("../negation-words.txt") as f:
            self.negationwords = [line.strip() for line in f]
        self.negationwords = set(self.negationwords)

        self.negative_word_count = 0 # Total number of words for each class
        self.positive_word_count = 0
        self.positive_words = {}  # Word frequencies for each class
        self.negative_words = {}

    def clean(self, text):        
        # Convert to lowercase
        message = text.lower()
        # Remove URLs. Match words that begin with http, http\S+, www, etc... and \S+ which is just one or more non-whitespace characters
        message = re.sub(r'\shttp\S+|\swww\S+|\shttps\S+', ' URL ', message, flags=re.MULTILINE)

        # Do they same as above but match with anything that ends with .com, .net, or website endings
        message = re.sub(r'\S+.com\s|\S+.net\s|\S+.org\s|\S+.co\s|\S+.us\s|\S+.edu\s|\S+.me\s|\S+.cn\s|\S+.uk\s|\S+.cn\s', ' URL ', message, flags=re.MULTILINE)
        # Get rid of @ mentions from the tweet dataset
        message = re.sub(r'@\S+', '', message, flags=re.MULTILINE)

        # Remove punctuation
        message = re.sub(r'\W', ' ', message)
        # Remove digits
        message = re.sub(r'\d+', '', message)

        # Tokenize the tweet (just store each word into a list)
        tokens = word_tokenize(message)

        tokens = [token for token in tokens if token not in self.stopwords] # Keep the ones that aren't a stop word

        # Lemmatize tokens, converting it back to its base form
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        tokens = [token for token in tokens if len(token) > 1] # Get rid of random letters out and about from punctuation removal
        return tokens # Return a list of tokens

    def fit(self, positive_text, negative_text):
        for i in range(len(positive_text)):
            splitted = positive_text[i].split()
            for j in range(len(splitted)):
                if splitted[j] not in self.positive_words: # Keep track of 
                    self.positive_words[splitted[j]] = 0
                self.positive_words[splitted[j]] += 1
                self.positive_word_count += 1

        for i in range(len(negative_text)):
            splitted = negative_text[i].split()
            for j in range(len(splitted)):
                if splitted[j] not in self.negative_words:
                    self.negative_words[splitted[j]] = 0
                self.negative_words[splitted[j]] += 1
                self.negative_word_count += 1
    
    def predict(self, text): # VERY similar to data preprocessing, 1. Preprocess 2. Run inference based off the words
        tokens = self.clean(text) # Clean the message and get tokens in the form that we want
        pos_word_prob = 1
        neg_word_prob = 1
        for i in range(len(tokens)):
            if tokens[i] in self.positive_words and tokens[i] in self.negative_words: # If this word is in both, we can simply grab it from both
                pos_word_prob *= self.positive_words[tokens[i]] / self.positive_word_count
                neg_word_prob *= self.negative_words[tokens[i]] / self.negative_word_count
            elif tokens[i] in self.positive_words: # If it's only in one of them, need to pretend we saw one of this word in the negative or positive dataset
                pos_word_prob *= self.positive_words[tokens[i]] / self.positive_word_count
                neg_word_prob *= 1 / (self.negative_word_count + len(self.negative_words)) # Divide by negative_word_count + number of words because we're pretending for each negative word, we saw more than we actually did to scale with the current word we are pretending exists
            elif tokens[i] in self.negative_words:
                neg_word_prob *= self.negative_words[tokens[i]] / self.negative_word_count
                pos_word_prob *= 1 / (self.positive_word_count + len(self.positive_words)) # Scale it up to the number of additions we need to "hallucinate"
        final_pos = pos_word_prob
        final_neg = neg_word_prob
        text_split = text.split()
        negate = False
        if len(text_split) > 2 and (text_split[0] in self.negationwords or text_split[1] in self.negationwords): # Use my list of negation words to detect if the first couple of words are a word that negates the rest
            negate = True
        if len(text_split) > 3 and text_split[2] in self.negationwords:
            negate = True
        if abs(0.5 - final_pos / (final_pos + final_neg)) < 0.05: # If there's not too much of a difference simply deem it neutral
            return 1
        if final_pos > final_neg: 
            if negate: # If we found a negation word at the start, chances are simply relying on the word's probability won't cut it so negate the outcome. Ex: Don't be harsh on yourself vs. Be harsh on yourself will be distuinguished with this
                return 0
            return 2
        else:
            if negate:
                return 2
            return 0    