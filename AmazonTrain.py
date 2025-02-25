import torch
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

class AmazonTrainSystem:
    def __init__(self, Dataset, ProcessedFeatures, Hyperparameters):
        Data = Dataset.GetData()
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Hyperparameters = Hyperparameters
        self.UseTuning = Hyperparameters["UseTuning"]

        self.XTrain, self.YTrain = ProcessedFeatures[Data['X_train'].index], Data['y_train']
        self.XVal, self.YVal = ProcessedFeatures[Data['X_val'].index], Data['y_val']
        self.XTest, self.YTest = ProcessedFeatures[Data['X_test'].index], Data['y_test']

        self.YTrain = self.YTrain - 1
        self.YVal = self.YVal - 1
        self.YTest = self.YTest - 1


        self.Models = {
            'LogisticRegression': LogisticRegression(C=Hyperparameters["LogisticC"], solver=Hyperparameters["LogisticSolver"]) if not self.UseTuning else LogisticRegression(max_iter=1000),
            'SVM': SVC(C=Hyperparameters["SVMC"], kernel='linear') if not self.UseTuning else SVC(kernel='linear'),
            'NaiveBayes': MultinomialNB(),
            'RandomForest': RandomForestClassifier(n_estimators=Hyperparameters["RFEstimators"]) if not self.UseTuning else RandomForestClassifier(n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(learning_rate=Hyperparameters["GBLearningRate"], n_estimators=Hyperparameters["GBEstimators"]) if not self.UseTuning else GradientBoostingClassifier(n_estimators=100),
            # 'LSTM': self.CreateLSTMModel(),  # Commented out due to execution time
        }

        self.Models['Ensemble'] = self.CreateEnsembleModel()
        self.ConvertDataForNaiveBayes()

    def ConvertDataForNaiveBayes(self):
        if 'NaiveBayes' in self.Models:
            self.XTrain_NB = self.XTrain.copy()
            self.XTest_NB = self.XTest.copy()
            self.XVal_NB = self.XVal.copy()
            self.XTrain_NB[self.XTrain_NB < 0] = 0
            self.XTest_NB[self.XTest_NB < 0] = 0
            self.XVal_NB[self.XVal_NB < 0] = 0

    def DataAugmentation(self):
        self.XTrain, self.YTrain = resample(self.XTrain, self.YTrain, replace=True, n_samples=int(len(self.XTrain) * 1.2), random_state=42)

    def Train(self):
        TrainingTimes = {}
        MemoryUsage = {}

        for Name, Model in self.Models.items():
            print(f"Training {Name}...")
            StartTime = time.time()
            StartMemory = psutil.virtual_memory().used / (1024 ** 3)

            if Name == 'NaiveBayes':
                Model.fit(self.XTrain_NB, self.YTrain)
            else:
                if self.UseTuning:
                    Model = self.HyperparameterTuning(Name, Model)
                Model.fit(self.XTrain, self.YTrain)

            EndTime = time.time()
            EndMemory = psutil.virtual_memory().used / (1024 ** 3)

            TrainingTimes[Name] = EndTime - StartTime
            MemoryUsage[Name] = EndMemory - StartMemory
            print(f"{Name} Done: {EndTime - StartTime:.2f}s, RAM: {EndMemory - StartMemory:.2f}GB")

        self.PlotTrainingStats(TrainingTimes, MemoryUsage)
        return self.Models

    def CreateEnsembleModel(self):
        return VotingClassifier(estimators=[
            ('LogisticRegression', self.Models['LogisticRegression']),
            ('SVM', self.Models['SVM']),
            ('RandomForest', self.Models['RandomForest']),
            ('GradientBoosting', self.Models['GradientBoosting'])
        ], voting='hard')

    def HyperparameterTuning(self, Name, Model):
        print(f"Tuning {Name}...")
        ParamGrid = self.Hyperparameters["GridSearchParams"].get(Name, {})
        if ParamGrid:
            GridSearch = GridSearchCV(Model, ParamGrid, cv=3, scoring='accuracy')
            GridSearch.fit(self.XTrain, self.YTrain)
            print(f"{Name} Best: {GridSearch.best_params_}")
            return GridSearch.best_estimator_
        return Model

    def PlotTrainingStats(self, TrainingTimes, MemoryUsage):
        """Plots training time and memory usage for each model"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(TrainingTimes.keys(), TrainingTimes.values(), color='blue')
        plt.xlabel("Models")
        plt.ylabel("Training Time (seconds)")
        plt.title("Training Time Per Model")

        plt.subplot(1, 2, 2)
        plt.bar(MemoryUsage.keys(), MemoryUsage.values(), color='red')
        plt.xlabel("Models")
        plt.ylabel("Memory Usage (GB)")
        plt.title("Memory Usage Per Model")

        plt.tight_layout()
        plt.show()
