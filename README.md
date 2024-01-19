# OVERVIEW OF THE REPOSITORY:

    First we preprocessed the conservations and the questions with the following functions
    def convert_to_lowercase(text):
    return text.lower()
    ```python
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
    Link different script and code pieces used and explain their function for the project

# METHODOLOGY:

    High-level explanation of things considered and solutions offered.

# RESULTS:

    Experimental findings supported by figures, tables etc.

# TEAM CONTRIBUTIONS

    List all team members by their names and how they contributed to the project
