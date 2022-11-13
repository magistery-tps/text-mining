import matplotlib.pyplot as plt
import seaborn           as sns


def pairplot(
    df, 
    hue,
    title   = '', 
    corner  = True,
    figsize = (20, 10)
):
    sns.set(rc={'figure.figsize' : figsize})
    plot = sns.pairplot(df, corner=corner, hue=hue)
    if title:
        plot.fig.suptitle(title, y=1.02)