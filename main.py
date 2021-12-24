from newspaper import Article
from newspaper import build
import pandas as pd
import os

def scrapping(url):
    articles = []
    urls_set = set()
    model = build(url, language = 'en', memoize_articles=False)
    for article in model.articles:
        if article.url and article.url not in urls_set:
            urls_set.add(article.url)
            articles.append(article.url)

    return articles

def extract_articles(articles, name):
    url = []
    content = []
    for article in articles:
        a = Article(article, language='en')
        a.download()
        a.parse()
        content.append(a.text)
        url.append(article)
    return [url, content]
    # df = pd.DataFrame(zip(url, content), columns=['url', 'content'])
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # folder = 'newspaper'
    # directory = os.path.join(dir_path, folder)
    # filename = name + '.pkl'
    # df.to_pickle(os.path.join(directory, filename))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    articles = scrapping('https://abcnews.go.com/')
    s = 'https://abcnews.go.com/'
    s = s[8:len(s)-1]
    extract_articles(articles, s)

    a = extract_articles(articles, s)
    url = a[0]
    content = a[1]



