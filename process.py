import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))
nltk.download('stopwords', quiet=True)
def preprocess(df):
    cleaned_texts = []  # Initialize a list to store preprocessed text
    #cleaned_text.clear()
    for text in df["text"]:
        # Removing characters other than letters and lowercasing
        review = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
        # Splitting into words
        review = review.split()
        # Applying Stemming and removing stopwords
        stemmed = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        # Joining words
        cleaned_text = ' '.join(stemmed)
        # Appending preprocessed text to cleaned_texts
        cleaned_texts.append(cleaned_text)
        stemmed.clear()
        review.clear()
    # Assign the list of preprocessed text to a new column in the DataFrame
    df['clean_text'] = cleaned_texts
    
    return df['clean_text']
