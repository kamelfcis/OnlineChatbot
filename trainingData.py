import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('questions.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Initialize lists to store words, classes, and training data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

# Iterate through each intent
for intent in intents['intents']:
    # Iterate through each pattern in the intent
    for pattern in intent['patterns']:
        # Tokenize the pattern into words
        word_list = nltk.word_tokenize(pattern)
        # Add words to the words list
        words.extend(word_list)
        # Add the document (pattern and tag) to the documents list
        documents.append((word_list, intent['tag']))
        # Add the tag to the classes list if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignore_letters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
# Sort and remove duplicates from words and classes
words = sorted(set(words))
classes = sorted(set(classes))

# Serialize words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Iterate through each document
for document in documents:
    # Initialize bag of words for the document
    bag = [0] * len(words)
    # Get the words from the document and lemmatize them
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Update bag of words for the words present in the document
    for word in word_patterns:
        if word in words:
            bag[words.index(word)] = 1

    # Create output row with one-hot encoding for the tag
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    # Append bag and output_row to training data
    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)
# Convert training data to numpy array
training = np.array(training, dtype=object)

# Split training data into input and output
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbotmodel.h5', hist)

# Print 'Done' after completion
print('Done')
