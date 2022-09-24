import metric as mt
from sklearn.metrics import classification_report


def plot_metrics(targets, predictions, figuresize=(8, 8)):
    print(classification_report(targets, predictions))
    mt.plot_confusion_matrix(
        targets, 
        predictions, 
        figuresize = figuresize
    )