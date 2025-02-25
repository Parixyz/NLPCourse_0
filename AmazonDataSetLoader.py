import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import numpy as np
import nlpaug.augmenter.word as naw
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class Processor:
    def __init__(self, preprocessingMode="lemmatization"):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.preprocessingMode = preprocessingMode

    def preprocess_text(self, text):
        if not isinstance(text, str):
            text = " ".join(text) if isinstance(text, list) else str(text)
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        if self.preprocessingMode == "stemming":
            tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        else:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def extract_features(self, text):
        sentiment_score = TextBlob(text).sentiment.polarity
        word_count = len(text.split())
        return sentiment_score, word_count

class AmazonDatasetLoader:
    def __init__(self, filename, augment_method=None, balance_method=None, preprocessingMode="lemmatization", trainRatio=1.0):
        self.filename = filename
        self.augment_method = augment_method
        self.balance_method = balance_method
        self.trainRatio = trainRatio
        self.processor = Processor(preprocessingMode)

        self.data = pd.read_csv(self.filename)

        required_columns = ['reviews.text', 'reviews.rating']
        for col in required_columns:
            if col not in self.data.columns:
                raise KeyError(f"Required column '{col}' is missing.")

        self.data = self.data[['reviews.text', 'reviews.rating']].dropna()
        self.data.columns = ['Text', 'Rating']

        self.plot_class_distribution("Before Augmentation")

        if self.augment_method:
            self.data = self.augment_data(self.data, method=self.augment_method)

        self.data['ProcessedText'] = self.data['Text'].apply(self.processor.preprocess_text)

#        if 'SentimentScore' not in self.data.columns or 'WordCount' not in self.data.columns:
#            self.data[['SentimentScore', 'WordCount']] = self.data['Text'].apply(lambda x: pd.Series(self.processor.extract_features(x)))

        if self.balance_method:
            self.data = self.balance_data(self.data, method=self.balance_method)

        if self.trainRatio < 1.0:
            self.data = self.data.sample(frac=self.trainRatio, random_state=42).reset_index(drop=True)

        self.plot_class_distribution("After Augmentation & Balancing")

        self.FeatureEngineering()

        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.data['ProcessedText'], self.data['Rating'], train_size=0.6, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def FeatureEngineering(self):
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        
        self.data['ProcessedText'] = self.data['ProcessedText'].astype(str)
        self.data['TFIDF'] = list(tfidf_vectorizer.fit_transform(self.data['ProcessedText'].tolist()).toarray())

        self.Word2VecModel = Word2Vec(sentences=[text.split() for text in self.data['ProcessedText']], vector_size=100, window=5, min_count=1, workers=4)
        self.data['Word2Vec'] = self.data['ProcessedText'].apply(lambda x: np.mean([self.Word2VecModel.wv[word] for word in x.split() if word in self.Word2VecModel.wv] or [np.zeros(100)], axis=0))

    def augment_data(self, data, method):
        if method == "synonym":
            aug = naw.SynonymAug(aug_p=0.3)
        elif method == "back_translation":
            aug = naw.BackTranslationAug()
        elif method == "random_insertion":
            aug = naw.RandomWordAug(action="insert")
        elif method == "contextual":
            aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
        else:
            return data

        augmented_texts = [aug.augment(text) for text in data['Text']]
        augmented_texts = [" ".join(text) if isinstance(text, list) else str(text) for text in augmented_texts]
        augmented_df = pd.DataFrame({'Text': augmented_texts, 'Rating': data['Rating']})
        return pd.concat([data, augmented_df]).reset_index(drop=True)

    def balance_data(self, data, method):
        if "ProcessedText" not in data.columns:
            return data  

        if method == "smote":
            tfidf = TfidfVectorizer(max_features=5000)
            X = tfidf.fit_transform(data['ProcessedText'])
            y = data['Rating']

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_data = pd.DataFrame({'ProcessedText': tfidf.inverse_transform(X_resampled), 'Rating': y_resampled})
            return balanced_data

        elif method == "oversampling":
            max_size = data['Rating'].value_counts().max()
            balanced_data = data.groupby('Rating').apply(lambda x: x.sample(max_size, replace=True)).reset_index(drop=True)
            return balanced_data

        elif method == "undersampling":
            min_size = data['Rating'].value_counts().min()
            balanced_data = data.groupby('Rating').apply(lambda x: x.sample(min_size, replace=False)).reset_index(drop=True)
            return balanced_data

        return data

    def plot_class_distribution(self, title):
        plt.figure(figsize=(8, 5))
        sns.countplot(x=self.data['Rating'], palette='coolwarm')
        plt.xlabel("Star Rating")
        plt.ylabel("Count")
        plt.title(title)
        plt.show()

    def Summary(self):
        print("\nDataset Summary:")
        print(f"Total Reviews: {len(self.data)}")
        print(f"Training Set Size: {self.X_train.shape[0]}")
        print(f"Validation Set Size: {self.X_val.shape[0]}")
        print(f"Testing Set Size: {self.X_test.shape[0]}")
        print(f"Missing Values: {self.data.isnull().sum().sum()}")

        if "Text" in self.data.columns:
            print(f"Average Review Length: {np.mean(self.data['Text'].apply(len)):.2f} characters")
        else:
            print("Warning: 'Text' column missing. Cannot calculate review length.")

    def GetData(self):
        return {'X_train': self.X_train, 'X_val': self.X_val, 'X_test': self.X_test,
                'y_train': self.y_train, 'y_val': self.y_val, 'y_test': self.y_test}

    def GetFullData(self):
        return self.data
