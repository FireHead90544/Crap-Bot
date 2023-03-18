import json
import nltk
from nltk import WordNetLemmatizer

class Setup:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.download_corpus()
        self.data = {}
        self.load_data()

    def load_data(self):
        with open("train_data.json", encoding="utf-8") as f:
            self.data = json.load(f)

    def download_corpus(self):
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download('omw-1.4')