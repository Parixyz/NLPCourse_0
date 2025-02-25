import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

class FeatureExtractor:
    def __init__(self, useTfidf=False, useWord2Vec=False, useNGrams=(1,2)):
        self.useTfidf = useTfidf
        self.useWord2Vec = useWord2Vec
        self.useNGrams = useNGrams
        self.vectorizer = TfidfVectorizer(ngram_range=self.useNGrams) if self.useTfidf else CountVectorizer(ngram_range=self.useNGrams)
        self.word2vec_model = None

    def fit_transform(self, data):
        if isinstance(data, np.ndarray):
            data = data.astype(str).tolist()
        elif isinstance(data, pd.Series):
            data = data.astype(str).tolist()

        transformed_data = self.vectorizer.fit_transform(data)

        if self.useWord2Vec:
            self.word2vec_model = Word2Vec(sentences=[text.split() for text in data], vector_size=100, window=5, min_count=1, workers=4)
            word2vec_features = np.array([np.mean([self.word2vec_model.wv[word] for word in text.split() if word in self.word2vec_model.wv] or [np.zeros(100)], axis=0) for text in data])
            return np.hstack((transformed_data.toarray(), word2vec_features))

        return transformed_data.toarray()

    def plot_top_features(self, dataset, top_n=20):
        feature_array = self.vectorizer.get_feature_names_out()
        feature_counts = self.vectorizer.transform(dataset.GetFullData()['ProcessedText']).sum(axis=0).A1
        feature_dict = dict(zip(feature_array, feature_counts))
        sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, counts = zip(*sorted_features)
        
        plt.figure(figsize=(10,5))
        sns.barplot(x=list(counts), y=list(words), palette='coolwarm')
        plt.xlabel('Frequency')
        plt.ylabel('Top Words')
        plt.title(f'Top {top_n} Features')
        plt.show()

    def plot_class_distribution(self, dataset, title="Class Distribution"):
        plt.figure(figsize=(8,5))
        sns.countplot(x=dataset['Rating'], palette='coolwarm')
        plt.xlabel("Star Rating")
        plt.ylabel("Count")
        plt.title(title)
        plt.show()