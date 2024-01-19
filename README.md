# CS412 Machine Learning Project

This repository is initiated for our machine learning course at Sabancı University. The collaborators of this project are Nusret Ali Kızılaslan, Ali Güneysu, Ebrar Berenay Yiğit, Kağan Kasapoğlu, and Mert Ziya. For the project, we tried to develop a machine learning model that deals with ChatGPT conversations by students, who consulted to it while they were solving their homework. The model is responsible to predict the expected score of the student, according to their ChatGPT conversations. The notebook and the material that we have used for the project can be found in this repository.

## Preprocessing
Preprocessing process was already done in the provided notebook. Therefore we additioally preprocessed the conservations and the questions with the following functions.
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

Additional to the keywords, we have also added some other features like ```turkish_characters``` and ```similarity_prompt_response```. When we skimmed through the HTML files, we have seen that some students communicated in Turkish language. ChatGPT is known to work best in English. Therefore, we thought that using Turkish language may be a sign of lower performance, resulting in lower grades. ```similarity_prompt_response``` could be a useful feature since it shows how the student asked question that are related to the responses. This could be an indicator how well the student dive deep into the questions and tried to solve the question properly. Therefore, this could be a classifier for high scores as well. We measured the similarity by Jaccard index.

```python
def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0
```

## Methodology

We have used several methods to deal with this project. We have used Decision Tree Classifiers, Random Forest, Support Vector Machines, Gradient Boosting, and Neural Network. Out of these 5 methods, the most promising methods were Gradient Boosting and Neural Network. Since the task is too complex to find a good classifier, we tried several configurations and various features to find the most optimal model. Due to the scaricity of data, we decided to use **cross-validation**. Initially, we used ```train_test_split``` but since there is not enough data to accurately test the model, cross-validation seemed to be the best way to do it. We have used 5 splits for K-fold. For the accuracy, we decided on threshold value of 6. If the difference between the threshold value and the predicted value is less than or equal to 6, it is considered as an accurate prediction. For each fold, we have calculated MSE (Mean Square Error), MAPE (Mean Absolute Percentage Error), and Accuracy. At the end, we have calculated these metrics across all folds. To see the overall accuracy of the model on the marginal rate, we ran the tests on whole data and plotted the results.


### Neural Network

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
This model consists of input layer, output layer, and two hidden layers. The model compiles with Adam optimizer with a low learning rate of 0.000625. We avoid making the model too complex since it would cause it to overfit. Dense layers are fully connected layers where every neuron is connected to each other. ReLU activation function is introduced for non-linearity. Since Dense layers are fully connected, there was a risk of overfitting. Adding ```Dropout``` layers with a rate of 0.5, helped us to avoid overfitting, as it drops out randomly selected neurons that may cause to overfit. Output layer is Dense layer with one neuron, with no acitvation function, which is typical for regression tasks where it outputs the predicted value directly. Epochs is set to 250 which gives the optimal accuracy for this task. Anything below 250 cause it to underfit and anything above cause it to overfit. 250 is just the right spot for it.

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

### Gradient Boosting

Gradient Boosting was our second best remedy for this task. At first, it seemed useful to use but after some re-testing and re-evaluating, we have came to a conclusion that it does not provide the precision that is required for this complex task. In our first trials, Gradient Boosting showed promising results, especially with cross-validation, until we tried with the brand new data that is provided by the instructor. Gradient Boosting model predicted huge majority of the entries as 100, which we believe that is not the case. We again used cross-validation and overall testing for the Gradient Boosting

```python
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
# Assuming 'X' contains your features and 'y' contains the target variable ('grade')
#Not using at the moment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Best parameters: {'learning_rate': 0.01, 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50, 'subsample': 0.8}
# Mean Squared Error on test set: 98.0395036389909
 

# Create and train the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=99, learning_rate=0.01, random_state=42, max_depth=3, subsample=0.8,min_samples_split=10, min_samples_leaf=4)

scores = cross_val_score(gb_model, X, y, cv=k_fold, scoring='neg_mean_squared_error')

print("Cross-Validation Scores:", scores)
print("Mean MSE:", -scores.mean())  # Convert back to positive for interpretation
print("Standard Deviation:", scores.std())

gb_model.fit(X, y)

# Make predictions on the test set
predictions_gb = gb_model.predict(X_test)

# Evaluate the Gradient Boosting model
mse_gb = mean_squared_error(y_test, predictions_gb)
print(f'Mean Squared Error (Gradient Boosting): {mse_gb}')

# Choose an index from X_test to test a specific data point
index_to_test = -6  # You can change this to the index you want to test

# Get the features of the specific data point
features_to_test = X_test[index_to_test, :].reshape(1, -1)

# Get the real target value for the specific data point
real_value = y_test[index_to_test]

# Make predictions for the specific data point
predicted_value = gb_model.predict(features_to_test)[0]

print(f"Real Value: {real_value}")
print(f"Predicted Value: {predicted_value}")
```

### SVM
We tried to work it out on SVM, but we did not see any potential after few trials so we abandon it. We did not use cross-validation for this one.

```python
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Assuming 'X' contains your features and 'y' contains the target variable ('grade')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVR model
model = SVR(kernel='linear')  # You can experiment with different kernels, such as 'linear', 'rbf', etc.
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
```
### Decision Tree

Same for this one. Did not dive well into it since the other models where much more promising.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Decision Tree Regressor model
tree_model = DecisionTreeRegressor(random_state=42)

# Fit the model to the training data
tree_model.fit(X_train, y_train)

# Predict the grades for the testing data
predictions_tree = tree_model.predict(X_test)

# Calculate the Mean Squared Error between the actual grades and the predicted grades
mse_tree = mean_squared_error(y_test, predictions_tree)
print(f'Mean Squared Error (Decision Tree): {mse_tree:.2f}')

# Assuming 'X' and 'y' are already defined and are a numpy array and a vector respectively
# Calculate the correlation matrix for the features and the target variable 'grade'
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['grade'] = y
correlation_matrix = df.corr()

# Identify the feature with the highest correlation (absolute value) to the grade
highest_correlation_feature = correlation_matrix['grade'].drop(labels=['grade']).abs().idxmax()
highest_correlation_value = correlation_matrix.loc[highest_correlation_feature, 'grade']

# Output the feature with the highest correlation to the grade
print(f'Highest correlation is with {highest_correlation_feature}: {highest_correlation_value:.3f}')
```

### Random Forest

Third most promising model. Frankly, as we already so busy with gradient boosting and neural network, we did not delve much into it. Here it is anyway.

```python

# Assuming 'X' contains your features and 'y' contains the target variable ('grade')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train the Random Forest model
forest_model = RandomForestRegressor()  # Experiment with different values


forest_model.fit(X_train, y_train)

# Make predictions on the test set
predictions_forest = forest_model.predict(X_test)

# Evaluate the Random Forest model
mse_forest = mean_squared_error(y_test, predictions_forest)
print(f'Mean Squared Error (Random Forest): {mse_forest}')

# Scatter plot of actual vs predicted values for RandomForest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions_forest, color='green', label='Actual vs Predicted (Random Forest)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
```



# RESULTS:

### Neural Network

#### Accuracy K-Fold

![8d93184c-8e7b-4be1-9662-5e7fd5e6406c](https://github.com/nusret35/cs412-project/assets/96892300/2e5e147f-ddb3-412f-ba70-b42f39436147)

**Fold 1**: Accuracy: 20.0%

**Fold 2**: Accuracy: 48.0%

**Fold 3**: Accuracy: 45.83333333333333%

**Fold 4**: Accuracy: 45.83333333333333%

**Fold 5**: Accuracy: 50.0%

**Mean Squared Error across 5 folds**: 229.9384558816477

**Mean Absolute Error across 5 folds**: 10.402523969014485

**Mean Absolute Percentage Error across 5 folds**: 16.301748181497338%

**Mean Accuracy across 5 folds**: 41.93333333333333%

#### Overall Performance

![0e9e9bde-4c1a-4cb9-9db6-597eec885d16](https://github.com/nusret35/cs412-project/assets/96892300/e3e1e748-199c-47a2-ac15-77991ff8d2fe)

**Mean Squared Error**: 97.01699171271058

**Mean Absolute Error**: 5.335291096421539

**Mean Accuracy**: 50.0

### GRADIENT BOOSTING 

### Cross Validation

![863947ad-912f-48f2-83fc-8fd1f946eb45](https://github.com/nusret35/cs412-project/assets/96892300/376a1a22-a9a5-484e-a23b-597f8590e99b)


#### Overall Performance

![788d10bc-c195-48f3-99b4-7219b1c12748](https://github.com/nusret35/cs412-project/assets/96892300/7f39e6f9-07df-4645-92cd-8e6f5e47344d)

Mean MSE: 159.98995135158026

**Standard Deviation**: 159.51154985209436

**Mean Squared Error (Gradient Boosting)**: 46.76876523513036


### SVM

#### Test Data Performance

![4ca201b7-27c5-44f2-a5f0-7290c3b1853b](https://github.com/nusret35/cs412-project/assets/96892300/4d69abb3-0fbe-4b72-a090-ee92b19fa5b2)

**Mean Squared Error**: 124.60196481037981

### Decision Tree

![aef6c4f9-9280-487e-90a1-a15f243bc3d5](https://github.com/nusret35/cs412-project/assets/96892300/85cb30e8-c04c-42a7-932f-8febb0cd3abc)



### Random Forest

#### Test Data Performance

![d48911fa-9424-4854-97af-2dfa8e0e3510](https://github.com/nusret35/cs412-project/assets/96892300/5d04b46b-f561-408c-97f9-fecda4483674)








# TEAM CONTRIBUTIONS:
 Our collaborative effort on this machine learning project involved a variety of tasks, and each team member played a crucial role. The following outlines the individual contributions of each team member:
 1. **Nusret Ali Kızılaslan**
 - Developed the neural network model.
 - Contributed to the selection and fine-tuning of hyperparameters for the neural network.
 - Participated actively in Zoom meetings to discuss the overall model architecture and its integration into the project.
 2. **Ali Vehbi Güneysu**
 - Developed the gradient boosting model.
 - Conducted extensive experimentation with hyperparameters to optimize the performance of the gradient boosting model.
 - Collaborated in Zoom meetings to share insights and findings related to the gradient boosting approach.
 3. **Ebrar Berenay Yiğit**
 - Developed the Support Vector Machine (SVM) model.
 - Contributed to the feature engineering process and extraction of relevant information from the ChatGPT conversations.
 - Actively participated in Zoom meetings to discuss SVM model results and fine-tune parameters.
 4. **Kağan Kasapoğlu**
 - Developed the random forest model.
 - Contributed to the feature engineering phase, identifying key keywords and patterns.
 - Engaged in collaborative decision-making during Zoom meetings, providing valuable input on model selection and performance.
 5. **Mert Ziya**
 - Developed the decision tree classifier model.
 - Contributed to the preprocessing of conversations and questions, ensuring data quality.
 - Actively participated in Zoom meetings, offering insights into decision tree model behavior and results.
 
 **Collective Efforts:**
 - Collaboratively worked on the development of the final machine learning model that decided as neural network model.
 - Engaged in frequent Zoom meetings to discuss progress, share insights, and collectively decide on the best approaches for the project.
 - Contributed to the preprocessing of data, including text cleaning, tokenization, and feature engineering.
 - Worked collectively to identify key features and insights from the data that could enhance the model's predictive capabilities.
 Our group synergy, facilitated by regular communication and shared responsibilities, enabled us to successfully develop and evaluate multiple machine learning models for the given task. The diverse expertise of each team member contributed to the comprehensive approach taken in tackling the challenges posed by the project.
