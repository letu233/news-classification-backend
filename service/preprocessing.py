import re
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def clean_train_data(x):
    text = str(x)
    text = re.sub('[0-9]+', '', text)
    text = text.lower().strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub('(@[a-zA-Z]+)', ' ', text)
    text = re.sub(r'[\.]+([a-zA-Z])', r'. \1', text)
    text = re.sub(r'[^\w][\.\,]+',' ',text)
    text = re.sub('\-([a-zA-Z]+)', r'\1', text)
    text = re.sub(r'[\_\-]', ' ', text)
    text = re.sub('ー', ' ', text)
    text = re.sub(r'[^\w\s]','',text)# remove punctuation
    text = re.sub(r'[^a-zA-Z]+',' ',text) # remove non alphabet
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub('\n+', ' ', text)
    text = re.sub('\s+', ' ', text)

    return text

def remove_stopwords(text):
    stop = {'these', "wouldn't", 'll', 'me', "haven't", '+', 've', "don't", "shouldn't", '_', 'to', 'them', 'down', '$', "you'd", 'into', ';', 'having', 'few', '&', ',', 'both', 'now', 't', 'where', 'which', 'during', 'doesn', 'your', 'from', 'same', '.', '*', 'needn', 'my', 'ain', 'ma', '`', 'too', 'ourselves', 'himself', 'but', 'while', 'when', 'out', 'all', "mustn't", '[', 'before', ')', 'by', 'its', 'or', 'that', 'above', "weren't", "shan't", 'this', 'an', 'couldn', '^', '%', 'theirs', 'wasn', 'off', 'itself', 'other', 'not', 'myself', 'mightn', "doesn't", 'own', "needn't", 'been', '>', 'you', 'as', '<', 'she', '{', "you've", 'be', 'the', 'because', 'who', 'are', 'of', 'should', 'isn', 'some', 'no', 'we', 'weren', 'won', 'after', "you'll", 'wouldn', 'their', 'most', 'will', 'again', "aren't", 'm', 'once', 'herself', "you're", 'have', 'doing', '|', 'don', 'had', 'if', '!', 'mustn', ']', 'each', '#', 'what', 'under', 'against', 'for', 'than', '-', 'being', 'themselves', "isn't", 'i', 'her', "she's", 'just', 'in', "mightn't", "hadn't", 'hers', 's', 'haven', '\\', 'more', 'our', '(', 'up', "that'll", 'until', 'about', "won't", "couldn't", "didn't", 'his', 'through', 'ours', '~', 'has', "it's", 'any', 'hadn', 'nor', ':', 'were', 'at', 'those', '/', 'and', 'him', 'then', "should've", "'", 'yourself', 'o', 'only', "wasn't", "hasn't", 'did', 'he', 'shan', 'a', 'didn', '"', 'on', 'can', 'very', 'such', 'further', 'below', 'they', 'shouldn', 'whom', 'over', 'with', 'why', 're', 'aren', 'is', '?', 'y', '}', 'does', 'it', 'am', 'do', 'd', 'so', 'hasn', 'yours', '=', 'here', 'was', '@', 'between', 'there', 'how', 'yourselves'}
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())

    return " ".join(final_text)

def denoise_text(text):
    text = clean_train_data(text)
    text = remove_stopwords(text)

    return text

def tokenize(text):
    #nltk.download('punkt')
    token = word_tokenize(text)
    return token

def lemmatizer(word_list):
    lemmatizer = WordNetLemmatizer()
    #nltk.download('wordnet')
    text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    print(text) #text
    return text
