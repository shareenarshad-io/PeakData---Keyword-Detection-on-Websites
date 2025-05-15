#Introduction:
#As a first step in solving this problem, we will load the provided CSV files using the Pandas library. The training CSV file contains 100 rows, with three columns: URL, doc_id, and a label. The test CSV file has 48 rows with two columns: URL and doc_id. The goal is to train a machine learning model that can predict a label for the documents provided in the test CSV based on the data that is available in the training CSV.

import pandas as pd

train_csv = pd.read_csv(filepath_or_buffer="train.csv")
print("Training set shape", train_csv.shape)
train_csv.head()

test_csv = pd.read_csv(filepath_or_buffer="test.csv")
print("Test set shape", test_csv.shape)
test_csv.head()

tumor_keywords = pd.read_csv(filepath_or_buffer="keyword2tumor_type.csv")
print("Tumor keywords set shape", tumor_keywords.shape)
tumor_keywords.head()

'''
We have 100 documents in the training set, and 48 in the test set. We have 32 documents that mention no tumor board (label = 1), 59 documents where a tumor board is mentioned, but we are not certain if it is the main focus of the page (label = 2), and 9 documents for which we are certain that they are dedicated to tumor boards.
'''

train_csv.groupby(by="label").size()

#Loading Data

def read_html(doc_id: int) -> str:
    with open(file=f"htmls/{doc_id}.html",
              mode="r",
              encoding="latin1") as f:
        html = f.read()
    return html


train_csv["html"] = train_csv["doc_id"].apply(read_html)

train_csv.sample(n=5, random_state=42)

import warnings

from bs4 import BeautifulSoup

warnings.filterwarnings(action="ignore")


def extract_html_text(html):
    bs = BeautifulSoup(markup=html, features="lxml")
    for script in bs(name=["script", "style"]):
        script.decompose()
    return bs.get_text(separator=" ")


train_csv["html_text"] = train_csv["html"].apply(extract_html_text)

train_csv.sample(n=5, random_state=42)

from gensim.parsing import preprocessing


def preprocess_html_text(html_text: str) -> str:
    preprocessed_text = preprocessing.strip_non_alphanum(s=html_text)
    preprocessed_text = preprocessing.strip_multiple_whitespaces(s=preprocessed_text)
    preprocessed_text = preprocessing.strip_punctuation(s=preprocessed_text)
    preprocessed_text = preprocessing.strip_numeric(s=preprocessed_text)

    preprocessed_text = preprocessing.stem_text(text=preprocessed_text)
    preprocessed_text = preprocessing.remove_stopwords(s=preprocessed_text)
    return preprocessed_text


train_csv["preprocessed_html_text"] = train_csv["html_text"].apply(preprocess_html_text)

train_csv.sample(n=5, random_state=42)

#Exploratory Data Analysis

import plotly.express as px
import plotly.offline as pyo

# set notebook mode to work in offline
pyo.init_notebook_mode(connected=True)

px.histogram(x=train_csv["preprocessed_html_text"].apply(len), title="Distribution of Text Length (Character Count)")

px.histogram(x=train_csv["preprocessed_html_text"].apply(lambda text: text.split(" ")).apply(len),
             title="Distribution of Text Length (Word Count)")

px.histogram(x=train_csv["preprocessed_html_text"].apply(lambda text: set(text.split(" "))).apply(len),
             title="Unique Words Count")

#Modeling

import random
import numpy as np
import tensorflow as tf

# set the random seeds
np.random.seed(42)
tf.random.set_seed(seed=42)

#Data Generators
class Pair(tf.keras.utils.Sequence):
    def __init__(self, dataframe: pd.DataFrame, labels: pd.Series, n_batch: int, batch_size: int):
        self.dataframe = dataframe
        self.labels = labels
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.all_classes = set(self.labels)
        self.anchor_groups = {}
        for target_class in self.all_classes:
            self.anchor_groups[target_class] = {
                "positive": self.dataframe[self.labels == target_class],
                "negative": self.dataframe[self.labels != target_class]
            }

    def __len__(self):
        return self.n_batch

    def __getitem__(self, item):
        pairs = []

        for i in range(int(self.batch_size / 2)):
            anchor_class = random.randint(1, 3)
            anchor_group = self.anchor_groups[anchor_class]["positive"]
            not_anchor_group = self.anchor_groups[anchor_class]["negative"]

            anchor = anchor_group.sample(n=1).iloc[0]
            positive = anchor_group.sample(n=1).iloc[0]
            negative = not_anchor_group.sample(n=1).iloc[0]

            pairs.append([anchor, positive, 1])
            pairs.append([anchor, negative, 0])

        random.shuffle(x=pairs)
        pairs = np.array(pairs)

        data_pairs = pairs[:, :2]
        targets = pairs[:, 2]

        return data_pairs, tf.convert_to_tensor(targets, dtype=np.float32)

    def get_support_set(self, sample_size: int = 1):
        support_set = {}
        for target_class in self.all_classes:
            support_set[target_class] = self.anchor_groups[target_class]["positive"].sample(n=sample_size)
        return support_set
    
#Model Definition
'''
Here, we define our model, as a siamese network. The model is a sequence of layers, starting with a TextVectorization layer. This layer accepts natural language (text) as input, and maps it to an integer sequence. At initialization time, we should provide a vocabulary of words for it to be able to map the words at prediction time.

Following the text vectorization layer, we implement three Dense layers, with two Dropout layers in between. Lastly, we apply a L2 normalization layer to penalize large weights.
'''

class SiameseNetwork(tf.keras.Model):
    def __init__(self, corpora: pd.Series):
        super(SiameseNetwork, self).__init__()
        self.vectorizer_layer: tf.keras.layers.TextVectorization = tf.keras.layers.TextVectorization(
            max_tokens=2000,
            output_mode="int",
            output_sequence_length=512
        )
        self.vectorizer_layer.adapt(corpora.values)
        self.encoder = tf.keras.Sequential(layers=[
            self.vectorizer_layer,
            tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu),
            tf.keras.layers.Lambda(function=lambda x: tf.math.l2_normalize(x, axis=1))
        ])
        self.encoding_distance = tf.keras.layers.Dot(axes=1)

    def __call__(self, inputs, *args, **kwargs):
        anchors, supports = inputs[:, 0], inputs[:, 1]
        anchors_encoded = self.encoder(anchors)
        supports_encoded = self.encoder(supports)
        return self.encoding_distance((anchors_encoded, supports_encoded))

    def predict_with_support_set(self, entry, support_set: dict):
        scores = {}
        for instance_class, texts in support_set.items():
            class_scores = ([self(np.array([entry, text]).reshape((-1, 2))) for text in texts])
            scores[instance_class] = tf.math.reduce_mean(class_scores)
        return max(scores, key=scores.get)
    
model = SiameseNetwork(corpora=train_csv["preprocessed_html_text"])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_csv["preprocessed_html_text"], train_csv["label"],
                                                      test_size=0.2,
                                                      random_state=42, stratify=train_csv["label"])

# training params
BATCH_SIZE = 64
N_BATCH = 100
# we instantiate training and validation data / pair generators
TRAIN_PAIR_GENERATOR = Pair(dataframe=X_train, labels=y_train, n_batch=N_BATCH, batch_size=BATCH_SIZE)
VALID_PAIR_GENERATOR = Pair(dataframe=X_valid, labels=y_valid, n_batch=N_BATCH, batch_size=BATCH_SIZE)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

history = model.fit(
    x=TRAIN_PAIR_GENERATOR,
    validation_data=VALID_PAIR_GENERATOR,
    epochs=10,
    callbacks=[early_stopping_callback],
    verbose=1
)
'''
Model Evaluation
Once we finish with the model training we can start evaluating the produced model. All training information is stored in the history object that is returned by the model.fit() method. In the plots below, we plot the model's training and validation accuracy and loss over the number of epochs.
'''

import plotly.graph_objects as go

figure = go.Figure()

figure.add_scatter(y=history.history["binary_accuracy"], name="Training Accuracy")
figure.add_scatter(y=history.history["val_binary_accuracy"], name="Validation Accuracy")

figure.update_layout(dict1={
    "title": "Model Accuracy During Training",
    "xaxis_title": "Epoch",
    "yaxis_title": "Accuracy"
}, overwrite=True)

figure.show()

figure = go.Figure()

figure.add_scatter(y=history.history["loss"], name="Training Loss")
figure.add_scatter(y=history.history["val_loss"], name="Validation Loss")

figure.update_layout(dict1={
    "title": "Model Loss During Training",
    "xaxis_title": "Epoch",
    "yaxis_title": "Loss"
}, overwrite=True)

figure.show()

y_pred = X_valid.apply(lambda text: model.predict_with_support_set(
    entry=text,
    support_set=TRAIN_PAIR_GENERATOR.get_support_set(7)
))

# build a classification report
from sklearn.metrics import classification_report

report = classification_report(y_true=y_valid, y_pred=y_pred, zero_division=0)
print(report)

#Prediction

test_csv["html"] = test_csv["doc_id"].apply(read_html)

test_csv["html_text"] = test_csv["html"].apply(extract_html_text)

test_csv["preprocessed_html_text"] = test_csv["html_text"].apply(preprocess_html_text)

test_csv["preprocessed_html_text"] = test_csv["html_text"].apply(preprocess_html_text)

# do inference
test_csv["predictions"] = test_csv["preprocessed_html_text"].apply(lambda text: model.predict_with_support_set(
    entry=text,
    support_set=TRAIN_PAIR_GENERATOR.get_support_set(sample_size=7)
))

test_csv.sample(n=5, random_state=42)

test_csv["predictions"].value_counts()

test_csv[["doc_id", "predictions"]]

'''
Answers to Questions
In this section, we provide answers to the questions that were posed at the beginning of the assignment.

How did you decide to handle this amount of data?

We have used data generators that dynamically load the data samples from disk. It would have been possible to load the entire data set into memory, given that it is relatively small.

How did you decide to do feature engineering?

We haven't used any feature engineering techniques per se, though we have spent some effort on data pre-processing, with steps like removing punctuation, multiple whitespaces, non-alphanumerical characters, etc.

How did you decide which models to try (if you decide to train any models)?

We've decided to use the Siamese Network model because it is very popular for this particular task (natural language processing, small data set, class imbalance). The choice of layers is also very common in the field: we intertwine dropout layers with dense layers, which have a decreasing number of units. Lastly, we apply L2 regularization to penalize any large weights.

How did you perform validation of your model?

Validation is automatically handled by the Tensorflow library, we just pass in a validation set. The validation set was obtained by splitting the provided data into the train (80%) and validation (20%) sets.

What metrics did you measure?

During training we measure binary accuracy. In the evaluation phase, we measure per-class precision, recall, and f1 scores on the validation set.

How do you expect your model to perform on test data (in terms of your metrics)?

We expect somewhat similar performance to the validation set, around 0.5-6 f1 score on label = 1, around 0.8 f1 score on label = 2, and we hope, f1 > 0 on label = 3.

How fast will your algorithm perform and how could you improve its performance if you would have more time?

Each epoch takes around 30s to execute. We can improve that if we were to run the model on GPUs.

How do you think you would be able to improve your algorithm if you would have more data?

Build a more complex model
Try different loss metrics
Use pre-trained models
What potential issues do you see with your algorithm?

It is very prone to overfitting, though this is almost certainly because of the small data set.
We have zero precision and recall on the label = 3 which is concerning and should be addressed somehow.
'''