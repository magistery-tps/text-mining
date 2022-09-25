import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import itertools


def plot_metrics(targets, predictions, figuresize=(8, 8)):
    print(classification_report(targets, predictions))
    plot_confusion_matrix(
        targets, 
        predictions, 
        figuresize = figuresize
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    title      = 'Confusion matrix',
    cmap       = plt.cm.Blues,
    labels     = None,
    figuresize = (5, 5),
    x_rotation = 45
):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figuresize)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    ax = plt.gca()

    thresh = cm.max() / 2.
    for row, col in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value      = cm[row, col]
        text_color = "white" if value > thresh else "black"
        plt.text(row, col, value, horizontalalignment="center", color=text_color)

    classes_count = pd.Series(y_true).value_counts().index.values
    tick_marks = np.arange(len(classes_count))
    plt.xticks(tick_marks, rotation=45)
    plt.yticks(tick_marks)

    if labels:
        used_labels = [labels[r] for r in y_true_pred_values(y_true, y_pred)]
        ax.set_yticklabels(used_labels)
        ax.set_xticklabels(used_labels)
    else:
        ax.set_xticklabels((ax.get_xticks() +1).astype(str))

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
