import plotly.express as px 


def plot_tree(df, title="Arbol de categorias", count_column='count', figure='sunburst', size=1800, font_size=16):
    
    path = [c for c in df.columns if c != count_column]
    
    if figure == 'sunburst':
        fig = px.sunburst(
            df, 
            path   = path, 
            values = count_column, 
            width  = size,
            height = size
        )
    elif figure == 'treemap':
        fig = px.treemap(
            df, 
            path   = path, 
            values = count_column, 
            width  = size,
            height = size
        )
    else:
         raise 'Invalod figure type!'

    fig.update_layout(title_text=title, font_size=font_size)
    fig.show()