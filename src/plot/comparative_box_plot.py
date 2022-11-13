import matplotlib.pyplot as plt
import seaborn           as sns


def comparative_box_plot(
    df,
    x,
    y,
    figsize     = (8, 6),
    leyend      = False, 
    points      = True, 
    point_color = 'black',
    point_alpha = 0.1
):
    sns.set(rc={'figure.figsize' : figsize})
    sns.boxplot(
        data     = df, 
        x        = x, 
        y        = y, 
        hue      = y, 
        dodge    = False,
        notch    = True, 
        showcaps = False
    )
    if points:
        sns.stripplot(
            data  = df, 
            x     = x, 
            y     = y,
            color = point_color,
            alpha = point_alpha
        )
    if not leyend:
        plt.legend([],[], frameon=False)
    plt.show()