import AmazonDataSetLoader
import AmazonFeatureExtractor
import AmazonTrain
import AmazonEvaluate

if __name__ == '__main__':
    HYPERPARAMETERS = {
        "UseTuning": False,
        "Epochs": 5,
        "BatchSize": 32,
        "LearningRate": 2e-5,
        "Patience": 2,
        "LogisticC": 1,
        "LogisticSolver": "liblinear",
        "SVMC": 1,
        "RFEstimators": 100,
        "GBLearningRate": 0.1,
        "GBEstimators": 100,
        "GridSearchParams": {
            'LogisticRegression': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
            'SVM': {'C': [0.1, 1, 10]},
            'NaiveBayes': {},
            'RandomForest': {'n_estimators': [50, 100, 200]},
            'GradientBoosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100]}
        }
    }

    # Load dataset
    Dataset = AmazonDataSetLoader.AmazonDatasetLoader(
        'amazon_reviews.csv',
        augment_method="synonym",
        balance_method="smote",
        preprocessingMode="lemmatization",
        trainRatio=0.001
    )

    # Display dataset summary
    Dataset.Summary()

    # Extract features using TF-IDF and Word2Vec
    FeatureExtractor = AmazonFeatureExtractor.FeatureExtractor(useTfidf=True, useWord2Vec=True, useNGrams=(1, 2))
    ProcessedFeatures = FeatureExtractor.fit_transform(Dataset.GetFullData()['ProcessedText'])

    # Train models
    Trainer = AmazonTrain.AmazonTrainSystem(Dataset, ProcessedFeatures, HYPERPARAMETERS)
    TrainedModels = Trainer.Train()

    # Evaluate models
    Evaluator = AmazonEvaluate.AmazonEvaluate(TrainedModels, Trainer.XTest, Trainer.YTest)
    EvaluationResults = Evaluator.Evaluate()

    # Display evaluation results
    for Model, Metrics in EvaluationResults.items():
        print(f"Model: {Model}")
        print(f"Accuracy: {Metrics['Accuracy']}")
        print(f"Confusion Matrix:\n{Metrics['ConfusionMatrix']}")
        print(f"Classification Report:\n{Metrics['ClassificationReport']}")
