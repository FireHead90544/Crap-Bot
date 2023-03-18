import string # Punctuations stuffs
import random # Getting random responses and shuffling
import nltk # Tokenizing words and lemmatizing
import numpy as np # Managing the arrays
from tensorflow.keras import Sequential # The actual model
from tensorflow.keras.layers import Dense, Dropout # Preventing overfitting
from tensorflow.keras.models import load_model as lm # Loading the saved model
from keras.optimizers import Adam # The optimizer for our model
from utils import Setup

class CrapBotModel:
    """
    Our really crap bot model.
    Always says the truth.
    One thing about this is a lie.
    """
    def __init__(self):
        self.setup = Setup()
        self.words = []
        self.labels = []
        self.patterns = []
        self.pattern_tags = []
        self.prepare_data()
        self.train_x = []
        self.train_y = []
        self.input_shape = None
        self.output_shape = None
        self.model = Sequential()

    def prepare_data(self):
        for intent in self.setup.data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                self.patterns.append(pattern)
                self.pattern_tags.append(intent["tag"])

            if intent["tag"] not in self.labels:
                self.labels.append(intent["tag"])

        self.words = sorted(set([self.setup.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in string.punctuation]))
        self.labels = sorted(set(self.labels))

    def create_train_data(self):
        training = []
        out = [0] * len(self.labels)

        for indx, pattern in enumerate(self.patterns):
            bag = []
            text = self.setup.lemmatizer.lemmatize(pattern.lower())
            for word in self.words:
                bag.append(1) if word in text else bag.append(0)
            out_row = list(out)
            out_row[self.labels.index(self.pattern_tags[indx])] = 1

            training.append([bag, out_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        self.train_x = np.array(list(training[:, 0]))
        self.train_y = np.array(list(training[:, 1]))

        self.input_shape = (len(self.train_x[0]),)
        self.output_shape = len(self.train_y[0])

    def prepare_model(self, epochs=2500):
        self.model.add(Dense(128, input_shape=self.input_shape, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.output_shape, activation="softmax"))

        adam = Adam(learning_rate=0.01, decay=1e-6)

        self.model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    def load_model(self, path):
        self.model = lm(path)

    def model_summary(self):
        return self.model.summary()

    def train_model(self, epochs=2500, verbose=1):
        self.model.fit(x=self.train_x, y=self.train_y, epochs=epochs, verbose=verbose)

    def save_model(self, path):
        self.model.save(path)