import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

class AmazonEvaluate:
    def __init__(self, Models, XTest, YTest):
        self.Models = Models
        self.XTest = XTest
        self.YTest = YTest

    def Evaluate(self):
        Results = {}
        Predictions = {}

        for Name, Model in self.Models.items():
            Predictions[Name] = Model.predict(self.XTest)
            Accuracy = accuracy_score(self.YTest, Predictions[Name])
            ConfusionMatrix = confusion_matrix(self.YTest, Predictions[Name])
            ClassificationReport = classification_report(self.YTest, Predictions[Name], output_dict=True)

            Results[Name] = {
                'Accuracy': Accuracy,
                'ConfusionMatrix': ConfusionMatrix,
                'ClassificationReport': ClassificationReport
            }

            print(f"\n🔹 {Name} Accuracy: {Accuracy:.4f}")
            self.PlotConfusionMatrix(ConfusionMatrix, Name)
            self.PlotClassificationReport(ClassificationReport, Name)

        EnsemblePredictions = self.EnsembleVoting(Predictions)
        EnsembleAccuracy = accuracy_score(self.YTest, EnsemblePredictions)
        print(f"\n🔥 Ensemble Model Accuracy (Majority Voting): {EnsembleAccuracy:.4f}")

        self.MisclassificationAnalysis(Predictions)
        return Results

    def EnsembleVoting(self, Predictions):
        CombinedPredictions = np.array(list(Predictions.values()))
        FinalPredictions = [Counter(CombinedPredictions[:, i]).most_common(1)[0][0] for i in range(CombinedPredictions.shape[1])]
        return np.array(FinalPredictions)

    def PlotConfusionMatrix(self, ConfusionMatrix, ModelName):
        plt.figure(figsize=(6,5))
        sns.heatmap(ConfusionMatrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {ModelName}')
        plt.show()

    def PlotClassificationReport(self, Report, ModelName):
        Classes = list(Report.keys())[:-3]  
        Precision = [Report[Cls]['precision'] for Cls in Classes]
        Recall = [Report[Cls]['recall'] for Cls in Classes]
        F1Score = [Report[Cls]['f1-score'] for Cls in Classes]

        X = np.arange(len(Classes))

        plt.figure(figsize=(10,5))
        plt.bar(X - 0.2, Precision, 0.2, label="Precision", color='blue')
        plt.bar(X, Recall, 0.2, label="Recall", color='green')
        plt.bar(X + 0.2, F1Score, 0.2, label="F1-Score", color='red')

        plt.xticks(X, Classes, rotation=45)
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title(f'Classification Report - {ModelName}')
        plt.legend()
        plt.show()

    def MisclassificationAnalysis(self, Predictions):
        for ModelName, Preds in Predictions.items():
            MisclassifiedIndices = np.where(Preds != self.YTest)[0]

            print(f"\n X Misclassification Analysis for {ModelName}:")
            for Index in MisclassifiedIndices[:5]:
                print(f"Actual: {self.YTest.iloc[Index]}, Predicted: {Preds[Index]}")
