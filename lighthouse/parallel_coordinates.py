import os
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('output/bulb.csv')
# Create a mapping from model names to numbers
map_fn = {model: i for i, model in enumerate(df['Model'].unique())}
columns = ['Model', 'Device', 'CPU Count', 'RAM (GB)', 'VRAM (GB)', 'Model Size (GB)', 'GPU Layers', 
           'Batch', 'Threads', 'Batch Threads', 'Sample Time (Tk/s)', 'Prompt Eval Time (Tk/s)', 
           'Eval Time (Tk/s)']

df = df[columns]

def create_dict_dimensions(df):
    dicts = []
    for col in df.columns:
        if df[col].dtypes == 'object':
            dicts.append(
                dict(tickvals=list(range(len(df[col].unique()))), 
                     label=col, 
                     values=df[col].map({model: i for i, model in enumerate(df[col].unique())}),
                     ticktext = df[col].unique()))
        else:
            dicts.append(
                dict(label = col,
                     values = df[col],
                     range = [df[col].min(), df[col].max()])
            )
    return dicts

fig = go.Figure(data=
    go.Parcoords(
        line=dict(color = df['Model'].map(map_fn)),
        dimensions = list(create_dict_dimensions(df)),
        labelangle = -45,
        labelfont =dict(
            family="Courier New, monospace",
            size=20),
            rangefont=dict(
            family="Courier New, monospace",
            size=15),
        tickfont=dict(
            family="Courier New, monospace",
            size=15),
        unselected = dict(line = dict(color = 'grey', opacity = 0)),
        #legend=...
    )
)


fig.update_layout(
    margin=dict(l=380, r=150, t=250, b=50),
    width=2000,
    height=1000,
)
fig.show()
os.path.join('images', 'ParCoords.png')
fig.write_image(os.path.join('images', 'ParCoords.png'))
fig.write_html(os.path.join('images', 'ParCoords.html'))