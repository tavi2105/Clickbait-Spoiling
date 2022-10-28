import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
#### Optional
# in caz ca dorim sa inlocuim tokenizator din CountVectorizer cu unul facut de noi care
# sa se ocupe si de lematizare
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
#
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        regex_num_ponctuation = '(\d+)|([^\w\s])'
        stopw = stopwords.words("english")
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)
                if not re.search(regex_num_ponctuation, t) and t not in stopw]

###
#prelucrare date test
# citim json si il stocam intr-un dataframe
df_test = pd.read_json("./validation.jsonl", lines=True)

# coloana tags contine moment array cu un singur element asa ca desfacem array-ul
df_test["tags"] = list(map(lambda x: x[0],df_test["tags"].tolist()))
# etichetam valorile din tags cu valori numerice
df_test["tags"] = df_test["tags"].apply(lambda x: 0 if x=='phrase' else 1 if x=='passage' else 2)

# desfacem array-ul de stringuri din coloana ce contine paragrafele
df_test["targetParagraphs"] = df_test["targetParagraphs"].apply(lambda x:"\n".join(x) )
print(df_test["tags"].value_counts())

# prelucram datele de antrenare repetand pasii de mai sus
df = pd.read_json("./train.jsonl", lines=True)
df["tags"] = list(map(lambda x: x[0],df["tags"].tolist()))
print(df["tags"].value_counts())

df["tags"] = df["tags"].apply(lambda x: 0 if x=='phrase' else 1 if x=='passage' else 2)
print(df["tags"].value_counts())

df["targetParagraphs"] = df["targetParagraphs"].apply(lambda x:"\n".join(x) )

# definim modelul
model = Pipeline(
    steps=[
        (# aici ii dam algoritmul ce se va ocupa cu prelucrarea textului, transformand-ul intr-un vector cu valori numerice
            "count_verctorizer",CountVectorizer(stop_words='english')
        ),
        (# aici precizam algoritmul ml ce dorim sa-l efectuam
            "naive_bayes", MultinomialNB()
       )
])
# antrenam modelul
model.fit(df['targetParagraphs'], df['tags'])
# testam cu datele de test
y_pred = model.predict(df_test['targetParagraphs'])
# calculam scorul
print('Accuracy:', accuracy_score(df_test["tags"], y_pred))


#SVM
model2 = Pipeline(
    steps=[
        (# aici ii dam algoritmul ce se va ocupa cu prelucrarea textului, transformand-ul intr-un vector cu valori numerice
            "count_verctorizer",CountVectorizer(stop_words='english')
        ),

     ('tfidf', TfidfTransformer()),
        ('clf', SVC(kernel='rbf'))
])

model2.fit(df['targetParagraphs'], df['tags'])
y_pred = model2.predict(df_test['targetParagraphs'])
print(accuracy_score(df_test["tags"], y_pred))

#Regresia Logistica
model3 = Pipeline(steps=[
    (
    # aici ii dam algoritmul ce se va ocupa cu prelucrarea textului, transformand-ul intr-un vector cu valori numerice
        "count_verctorizer", CountVectorizer(stop_words='english')
    ),
    ('select', SelectKBest(k=4)),
    ('clf', LogisticRegression())])

model3.fit(df['targetParagraphs'], df['tags'])
y_pred = model3.predict(df_test['targetParagraphs'])
print(accuracy_score(df_test["tags"], y_pred))