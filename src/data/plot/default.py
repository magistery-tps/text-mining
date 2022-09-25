from matplotlib import pyplot as plt
import seaborn  as sns


def plot_defaults(
    xlabel         = '',
    ylabel         = '',
    title          = '',
    title_fontsize = 20,
    axis_fontsize  = 16,
    x_rotation     = 60,
    figsize        = (15, 6)
):
    plt.xticks(rotation=x_rotation)
    sns.set(rc={'figure.figsize': figsize})
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)
    plt.show()

    
def plot_count(ds, column, figsize=(10,4), title=''):
    df = ds.groupby(column) \
        .family \
        .count() \
        .reset_index(name="Count") \
        .sort_values(by='Count', ascending=False)

    sns.set_theme(style="whitegrid")
    sns.set(rc={"figure.figsize":figsize})
    ax = sns.barplot(data=df, x="Count", y=column)
    plot_defaults(xlabel="Count", ylabel=column, title=title, figsize=figsize)
    

def bi_bar_plot(df, y, hue, figsize=(8,5), horizontal=False, title=''):
    sns.set_theme(style="whitegrid")
    sns.set(rc={"figure.figsize":figsize})
    if horizontal:
        sns.barplot(data=df, y='count',x=y, hue=hue)
        plot_defaults(xlabel=y, ylabel='count', title=title, figsize=figsize)
    else:
        sns.barplot(data=df, x='count',y=y, hue=hue)
        plot_defaults(xlabel='count', ylabel=y, title=title, figsize=figsize)