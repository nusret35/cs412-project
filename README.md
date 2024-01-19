# CS412 Machine Learning Project

This repository is initiated for our machine learning course at Sabancı University. The collaborators of this project are Nusret Ali Kızılaslan, Ali Güneysu, Berenay Yiğit, Kağan Kasapoğlu, and Mert Ziya. For the project, we tried to develop a machine learning model that deals with ChatGPT conversations by students, who consulted to it while they were solving their homework. The model is responsible to predict the expected score of the student, according to their ChatGPT conversations. The notebook and the material that we have used for the project can be found in this repository.

## Preprocessing
First we preprocessed the conservations and the questions with the following functions
```python
def convert_to_lowercase(text):
    return text.lower()

def remove_special_characters(text):
    special_characters_to_be_removed = ["'", ",", "*", "_", "(", ")", "/", "&", "%", "+", "^", ";", "=", "\\", "-", "%",'"',".",":",">","?","!"]    
    cleaned_text = ''
    # Removing special characters from the text
    for char in text:
        if char in special_characters_to_be_removed:
            cleaned_text += ' '
        else:
            cleaned_text += char


    return cleaned_text

def tokenize(text):
    return word_tokenize(text) 


def remove_stopwords(wordlist):
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in wordlist if word.lower() not in stop_words]
    return filtered_words

def stemming(wordlist):
    stemmer = SnowballStemmer("english")
    stemmed_words = [stemmer.stem(word) for word in wordlist]
    return stemmed_words

def convert_list_to_string(wordlist):
    review = ""
    for word in wordlist:
        review = review + word + " "
    return review

def clean_text(text):
    # Lowercase
    cleaned = convert_to_lowercase(text)

    # Remove punctuation and special characters
    cleaned = remove_special_characters(cleaned)

    wordlist = tokenize(cleaned)

    # Stopword removal
    wordlist = remove_stopwords(wordlist)
    
    # Stemming
    wordlist = stemming(wordlist)

    cleaned_text = convert_list_to_string(wordlist)

    return cleaned_text
```
## Feature Engineering

For the features, we have added several keywords on top of the already given keywords. We added ```impurity```, ```gain```, ```hyperparameter```, ```sure```, ```understand```,and ```please```. These keywords seem to have some insight about the student's approach. For example, "impurity" and "gain" are essential keywords, especially when the student is dealing with the information gain question. This feature may classify the ones that got higher scores. The words like "please" and "thank" seem to be used mostly by the people who are not very familiar with ChatGPT. So, these keywords may classify the ones that got lower scores. 

Additional to the keywords, we have also added some other features like ```turkish_characters``` and ```similarity_prompt_response```. When we skimmed through the HTML files, we have seen that some students communicated in Turkish language. ChatGPT is known to work best in English. Therefore, we thought that using Turkish language may be a sign of lower performance, resulting in lower grades. ```similarity_prompt_response``` could be a useful feature since it shows how the student asked question that are related to the responses. This could be an indicator how well the student dive deep into the questions and tried to solve the question properly. Therefore, this could be a classifier for high scores as well.

## Methodology

We have used several methods to deal with this project. We have used Decision Tree Classifiers, Random Forest, Support Vector Machines, Gradient Boosting, and Neural Network. Out of these 5 methods, the most promising methods were Gradient Boosting and Neural Network. Since the task is too complex to find a good classifier, we tried several configurations and various features to find the most optimal model. Due to the scaricity of data, we decided to use **cross-validation**. Initially, we used ```train_test_split``` but since the there is not enough data to accurately test the model, cross-validation seemed to be the best way to do it.

### Neural Netowrok

For neural network we have build the following model:

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

# Assuming 'X' contains your features and 'y' contains the target variable ('grade')

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build the neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model with a lower learning rate
model.compile(optimizer= Adam(learning_rate=0.001), loss='mean_squared_error')

# Number of splits for cross-validation
n_splits = 5

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_values = []
mae_values = []
mape_values = []
accuracy_values = []

# Perform cross-validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.1, verbose=0)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
    
    # Calculate accuracy (adjust threshold as needed)
    threshold = 5  # Adjust as needed for your specific problem
    correct_predictions = np.sum(np.abs(predictions.flatten() - y_test) < threshold)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions * 100

    mse_values.append(mse)
    mae_values.append(mae)
    mape_values.append(mape)
    accuracy_values.append(accuracy)

    print(f'MSE: {mse}, MAE: {mae}, MAPE: {mape}%, Accuracy: {accuracy}%')

# Calculate the mean values across all folds
mean_mse = np.mean(mse_values)
mean_mae = np.mean(mae_values)
mean_mape = np.mean(mape_values)
mean_accuracy = np.mean(accuracy_values)

print(f'Mean Squared Error across {n_splits} folds: {mean_mse}')
print(f'Mean Absolute Error across {n_splits} folds: {mean_mae}')
print(f'Mean Absolute Percentage Error across {n_splits} folds: {mean_mape}%')
print(f'Mean Accuracy across {n_splits} folds: {mean_accuracy}%')

```



# RESULTS:

    Experimental findings supported by figures, tables etc.

# TEAM CONTRIBUTIONS

    List all team members by their names and how they contributed to the project

