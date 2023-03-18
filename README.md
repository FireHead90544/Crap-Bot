# Crap-Bot
A deep learning-powered chatbot written in python which just works.

## About
An open source release of the chat bot based on the Sequential deep learing model we created to be deployed on our [college's website](https://ittanakpur.ac.in/) which itself need and is getting serious updates pretty soon. It was named Crap-Bot because why not? Because, we just felt like doing that, no definite reasons, we just felt good, no good lying.

## Developers
- [Rudransh Joshi](https://github.com/FireHead90544) - [B.Tech 1st Year (AI/ML)] - Core Developer & Dataset Collection
- [Uttam Tiwari](#) - [B.Tech 1st Year (CSE)] - Dataset Collection & Training
- [Neha Kumari](#) - [B.Tech 1st Year (CSE)] - Dataset Collection & Bug Testing
- [Rahul Verma](#) - [B.Tech 1st Year (CSE)] - Bug Testing & Frontend Development

The bot will soon be deployed on our college's website and will be available for everyone to use.

## Usage
To use this bot, you need to have python3 installed on your system. This bot was created and tested against python 3.11.0, but it should work with any python 3 version. Follow the instructions below to get started.

1. Clone the repository and cd into the directory.
```bash
git clone https://github.com/FireHead90544/Crap-Bot
cd Crap-Bot
```

2. Create a virtual environment.
It's a good practice to create a virtual environment for each project. This will keep your system's python installation clean and will prevent any conflicts with other projects. To create a virtual environment install `virtualenv` using pip (`pip install virtualenv`), and run the following command.
```bash
virtualenv venv
```

3. Activate the virtual environment.
```bash
source venv/bin/activate # Linux
.\venv\Scripts\activate # Windows
```

4. Install the required dependencies.
```bash
pip install -r requirements.txt
```

5. Prepare your dataset.
The bot uses a dataset to train itself. The dataset should be a json file named `train_data.json` which contains the training data, in a particular format. A sample `train_data.json` can be found in the repository.

Once you have done this, you can start training/testing the bot.


## Training
To train the bot, create a python file and write the below equivalent code and run it.
```py
from train import CrapBotModel

model = CrapBotModel() # Instantiate the crap bot

model.create_train_data() # Prepares the data in numeric form as arrays.
model.prepare_model() # Prepares the actual Sequential model for training.

print(model.model_summary()) # Shows the summary of the model about the neural network formed.

model.train_model(epochs=2000) # Trains the model, epochs represents the number of steps the model will train itself.
model.save_model("weights.h5") # Saves the model to a file.
```
This will train the model using the given dataset and save the model file to `weights.h5`. You can change the name of the file to whatever you want and load it later to retrain it or to test it out to make predictions.

## Re-training / Fine-tuning
To retrain the model, you can load the model file and train it again. The code below shows how to do that.
```py
from train import CrapBotModel

model = CrapBotModel() # Instantiate the crap bot

model.load_model("weights.h5") # Load the model file named 'weights.h5'
model.create_train_data() # Prepares the updated data from `train_data.json` in numeric form as arrays. You can add new data to the json file and retrain the model without having to create the model again.

print(model.model_summary()) # Shows the summary of the model about the neural network formed.

model.train_model(epochs=500) # Trains the model, epochs represents the number of steps the model will train itself.
model.save_model("weights2.h5") # Saves the model to a file.
```
This will re-train or fine-tune the previously trained model with the new added data to a file named `weights2.h5` without having to create the model and train it from scratch. It basically reduces the time required to train the new model instead of creating it from scratch since it trains over the already trained model, thus making it more accurate.

## Testing
To test the model, you can load the model file and test it. The code below shows how to do that.
```py
from test import Predict

predict = Predict("weights.h5") # Loads the model file and instantiates the prediction.
msg = "Hello."
predict.chat(msg) # Predicts the response to the message.
```
This will load the model file named `weights.h5` and predict the response to the message `Hello.`. You can change the message to whatever you want and test it out.

## License
This is licensed under the [GNU GPL v3](https://raw.githubusercontent.com/FireHead90544/Crap-Bot/main/LICENSE) license.


**Dear crappy peoples, enjoy crapping your crap-talks with the crap-bot!**