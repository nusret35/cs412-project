# OVERVIEW OF THE REPOSITORY:


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

# METHODOLOGY:

    High-level explanation of things considered and solutions offered.

# RESULTS:

    Experimental findings supported by figures, tables etc.

# TEAM CONTRIBUTIONS

    List all team members by their names and how they contributed to the project

