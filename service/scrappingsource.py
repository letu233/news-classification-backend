from newspaper import build
from newspaper import Article
import pandas as pd
import pickle
from joblib import load
import os
from nltk import word_tokenize
from newspaper import Config
from service.preprocessing import denoise_text, lemmatizer


def scrappingSources(list_url, ver):
    #Articles config
    config = Config()
    config.memoize_articles = False
    config.fetch_images = False
    config.MIN_WORD_COUNT = 400
    config.MIN_SENT_COUNT = 10

    for url in list_url:
        articles = []
        urls_set = set()
        news_articles = build(url, config=config)
        for article in news_articles.articles:
            if article.url and article.url not in urls_set:
                urls_set.add(article.url)
                articles.append(article.url)

        link = []
        content = []
        for article in articles:
            a = Article(article, language='en')
            try:
                a.download()
                a.parse()
            except:
                pass
            content.append(a.text)
            link.append(article)

        df = pd.DataFrame(zip(link, content), columns=['link', 'content'])
        df = df[df['content'].map(lambda x: len(str(x)) > 150)]

        filename = map_source(url)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        save_folder = os.path.join(dir_path, "articles")
        save_version = os.makedirs(os.path.join(save_folder, ver))
        save_path = os.path.join(os.path.join(save_folder, ver))
        save_file = os.path.join(save_path, filename)
        df.to_pickle(save_file)
        print(filename+ " saved!")


def predictSources(ver):
    vectorizer = pickle.load(open("vectorizer.pickle", 'rb'))
    loaded_model = load("xgboost_model.joblib.dat")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_folder = os.path.join(dir_path, "articles")
    save_files = os.path.join(save_folder, ver)
    dirs = os.listdir(save_files)

    for dir in dirs:
        # load and preprocess data
        file = os.path.join(save_files, dir)
        df = pd.read_pickle(file)
        df['content'] = df['content'].apply(denoise_text)
        df['content'] = df['content'].apply(lambda x: word_tokenize(str(x)))
        df['content'] = df['content'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
        df['content'] = df['content'].apply(lambda x: ' '.join(x))

        # transform using tf-idf
        test = vectorizer.transform(df.content)
        fake = []
        real = []
        for i in range(len(df)):
            res = loaded_model.predict_proba(test[i])
            fake.append(res[0][0])
            real.append(res[0][1])
        df['fake'] = fake
        df['real'] = real
        del df['content']
        df['fake'] = df['fake'].apply(lambda x: x * 100)
        df['real'] = df['real'].apply(lambda x: x * 100)
        df['fake'] = df['fake'].round(2)
        df['real'] = df['real'].round(2)

        #save dataframe
        save_name = "predict_" + dir
        save_tofolder = os.path.join(dir_path, "result")
        save_dir = os.makedirs(os.path.join(save_tofolder, ver))
        save_ver = os.path.join(save_tofolder, ver)
        df.to_pickle(os.path.join(save_ver, save_name))
        print(save_name+ " saved!")

def map_source(url):
  switcher = {
      'https://abcnews.go.com/': 'abcnews.pkl',
      'https://www.cnbc.com/world/?region=world': 'cnbc.pkl',
      'https://www.foxnews.com/': 'foxnews.pkl',
      'https://www.reuters.com/': 'reuters.pkl',
      'https://www.theguardian.com/international': 'guardian.pkl',
      'https://www.usatoday.com/': 'usatoday.pkl',
      'https://edition.cnn.com/': 'cnn.pkl'
  }
  return switcher.get(url, "invalid")

