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

We have used several methods to deal with this project. We have used Decision Tree Classifiers, Random Forest, Support Vector Machines, Gradient Boosting, and Neural Network. Out of these 5 methods, the most promising methods were Gradient Boosting and Neural Network. Since the task is too complex to find a good classifier, we tried several configurations and various features to find the most optimal model. Due to the scaricity of data, we decided to use **cross-validation**. Initially, we used ```train_test_split``` but since the there is not enough data to accurately test the model, cross-validation seemed to be the best way to do it. We have used 5 splits for K-fold. For the accuracy, we decided on threshold value of 6. If the difference between the threshold value and the predicted value is less than or equal to 6, it is considered as an accurate prediction. For each fold, we have calculated MSE (Mean Square Error), MAPE (Mean Absolute Percentage Error), and Accuracy. At the end, we have calculated these metrics across all folds. To see the overall accuracy of the model on the marginal rate, we ran the tests on whole data and plotted the results.

### Neural Netowrok

For neural network we have build the following model:

```python
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
model.compile(optimizer= Adam(learning_rate=0.000625), loss='mean_squared_error')
```
This model consists of input layer, output layer, and two hidden layers. The model compiles with Adam optimizer with a low learning rate of 0.000625. We avoid making the model too complex since it would cause it to overfit. Also adding ```Dropout``` layers with a rate of 0.5, helped us to avoid overfitting, as it drops out randomly selected neurons that may cause to overfit. 

### Cross Validation

Cross validation consisted with 5 splits. We fit the model for every split and take the record of that particular iteration. We set epochs to 250, batch size to 32, validation split to 0.1. These values were set after several trials and we believe we finally found the sweet-spot for the optimal results. 

```python
    # Number of splits for cross-validation
    n_splits = 5

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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
        threshold = 6  # Adjust as needed for your specific problem
        correct_predictions = np.sum(np.abs(predictions.flatten() - y_test) < threshold)
        total_predictions = len(y_test)
        accuracy = correct_predictions / total_predictions * 100
    
        mse_values.append(mse)
        mae_values.append(mae)
        mape_values.append(mape)
        accuracy_values.append(accuracy)
    
        print(f'MSE: {mse}, MAE: {mae}, MAPE: {mape}%, Accuracy: {accuracy}%')
```
### Overall Validation

Addition to cross validation, we tested the model on the whole data. This validation showed us how well our model performs in overall. This validation was also beneficial to decide whether the model is underfitted or overfitted.
```python
for index_to_predict in indices_to_predict:
    # Access the specific input from the test set
    input_to_predict = X[index_to_predict, :]

    # If you used scaling during training, apply the same scaling to the input
    input_to_predict_scaled = scaler.transform(input_to_predict.reshape(1, -1))

    prediction = model.predict(input_to_predict_scaled)

    # Append actual and predicted values for later error calculation
    y_actual.append(y[index_to_predict])
    y_predicted.append(min(prediction[0][0], 100))  # Ensure the prediction is within [0, 100]

    print(f"Actual Grade: {y[index_to_predict]}")
    print(f"Predicted Grade: {min(prediction[0][0], 100)}")

    threshold = 6

    correct_predictions = np.sum(np.abs(predictions.flatten() - y_test) < threshold)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions * 100
    accuracy_values.append(accuracy)
```


# RESULTS:

    Experimental findings supported by figures, tables etc.

# TEAM CONTRIBUTIONS

    List all team members by their names and how they contributed to the project

