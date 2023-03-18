from train import CrapBotModel
from numpy import array
import nltk
import random

class Predict():
    def __init__(self, model_path):
        self.model = CrapBotModel()
        self.model.load_model(model_path)

    def chat(self, query):
        intents = self.prediction_label(query, self.model.words, self.model.labels)
        return self.get_response(intents, self.model.setup.data)

    def clean_text(self, text):
        return [self.model.setup.lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]

    def bag_of_words(self, text, vocab):
        tokens = self.clean_text(text)
        bag = [0] * len(vocab)
        for w in tokens:
            for indx, word in enumerate(vocab):
                if word == w:
                    bag[indx] = 1

        return array(bag)

    def prediction_label(self, text, vocab, labels):
        bag = self.bag_of_words(text, vocab)
        result = self.model.model.predict(array([bag]), verbose=0).tolist()[0]
        threshold = 0.2
        y_prediction = [[indx, r] for indx, r in enumerate(result) if (r > threshold)]
        y_prediction.sort(key=lambda x: x[1], reverse=True)
        final = []
        for item in y_prediction:
            final.append(labels[item[0]])
        return final

    def get_response(self, intents, train_data):
        tag = intents[0]
        list_of_intents = train_data["intents"]
        for intent in list_of_intents:
            if intent["tag"] == tag:
                result = random.choice(intent["responses"])
                break
        return result