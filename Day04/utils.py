from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

stopword = set(stopwords.words('english'))

remove = ['no', 'never', 'nor', 'None', 'not']

for w in remove:
    if w in stopword:
        stopword.remove(w)

lemmatizer = WordNetLemmatizer()

def text_clean(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()

    # Tokenization
    tokens = text.split()

    # Stopword removal and lemmatization
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stopword
    ]

    return " ".join(tokens)
