import plotly.express as px 


def plot_tree(df, title="Arbol de categorias", path=['source', 'target'], figure='sunburst', width=1000, height=1000, font_size=15):
    if figure == 'sunburst':
        fig = px.sunburst(df, path=path,width=width, height=height)
    elif figure == 'treemap':
        fig = px.treemap(df, path=path,width=width, height=height)
    else:
         raise 'Invalod figure type!'

    fig.update_layout(title_text=title, font_size=font_size)
    fig.show()