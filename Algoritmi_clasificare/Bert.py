import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction import DictVectorizer

#prelucrare date test(la fel ca la naive bayes)
df_test = pd.read_json("./validation.jsonl", lines=True)
df_test["tags"] = list(map(lambda x: x[0],df_test["tags"].tolist()))
df_test["tags"] = df_test["tags"].apply(lambda x: 0 if x=='phrase' else 1 if x=='passage' else 2)
df_test["targetParagraphs"] = df_test["targetParagraphs"].apply(lambda x:"\n".join(x) )
#prelucrare date antrenament
df = pd.read_json("./train.jsonl", lines=True)
df["tags"] = list(map(lambda x: x[0],df["tags"].tolist()))
df["tags"] = df["tags"].apply(lambda x: 0 if x=='phrase' else 1 if x=='passage' else 2)
df["targetParagraphs"] = df["targetParagraphs"].apply(lambda x:"\n".join(x) )

#download two models, one to perform preprocessing and the other one for encoding.
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

#Initializing the BERT layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

#Initializing the neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# definire model
model = tf.keras.Model(inputs=[text_input], outputs = [l])

# metricele trebuie schimbate cu ceva mai specific noua
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]
# si la fel si aici
model.compile(optimizer='adam',
 loss='binary_crossentropy',
 metrics=METRICS,
              run_eagerly=True)

# antrenare (la epoci am pus doar unul ca sa imi ruleze mai repede dar noi ar trebui sa avem macar 10)
model.fit(df['targetParagraphs'], df['tags'], epochs=1)
#predictie
y_predicted = model.predict(df_test['targetParagraphs'])
y_predicted = y_predicted.flatten()
y_predicted = np.where(y_predicted > 0.5, 1, 0)

#afisare scor
print(accuracy_score(y_predicted,df_test['tags']))