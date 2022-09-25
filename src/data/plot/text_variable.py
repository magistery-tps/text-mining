import plotly.express as px 
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def words_clous_plot(
    words,
    min_font_size    = 7,
    random_state     = 21,
    max_font_size    = 50,
    relative_scaling = 0.5,
    colormap         = 'Dark2',
    title            = '',
    title_fontsize   = 20
):
    if type(words) != list:
        raise Exception("Words arg must be a strings list!")

    if len(words) == 0:
        return

    if type(words[0]) != str:
        raise Exception("Words arg must be a strings list!")

    text = " ".join(words)

    word_cloud = WordCloud(
        min_font_size    = min_font_size,
        random_state     = random_state,
        max_font_size    = max_font_size,
        relative_scaling = relative_scaling,
        colormap         = colormap
    ).generate(text)

    # Display the generated Word Cloud
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=title_fontsize)
    plt.show()



def describe_text_var(df, column, flatten=False):
    values = df[column].values.tolist()

    if flatten:
        values = [vv for v in values for vv in v]

    words_clous_plot(values, title = f'{column} text variable')