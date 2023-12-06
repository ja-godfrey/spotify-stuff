# %%
# https://chat.openai.com/c/9f85b511-eb36-4216-bde6-29eb2d1a9be5
threshold = 0.4

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import colorsys

# Load your dataset
df = pd.read_csv('./../data/derived/a-combined.csv')

# columns_to_analyze = [
#     'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 
#     'Liveness', 'Valence', 'Tempo', 'Danceability', 'Energy', 'Popularity', 'Duration (ms)'
# ]

columns_to_analyze = [
    'Duration (ms)', 'Popularity', 'Danceability', 'Energy', 'Key', 'Loudness','Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Time Signature'
                      ]
df_unique = df.drop_duplicates(subset='Spotify ID')[columns_to_analyze]

# Create a color map
def generate_pastel_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        # Adjust saturation (0.4-0.6) and lightness (0.7-0.9) for pastel tint
        colors.append(colorsys.hsv_to_rgb(hue, 0.5, 0.8))
    return colors

def to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))

unique_owners = df['PlaylistOwner'].unique()
n_owners = len(unique_owners)
pastel_colors = generate_pastel_colors(n_owners)
colors_hex = [to_hex(color) for color in pastel_colors]
color_map = dict(zip(unique_owners, colors_hex))

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_unique)

similarity_matrix = cosine_similarity(df_scaled)

G = nx.Graph()
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):
        # You can set a threshold for similarity to reduce the number of edges
        if similarity_matrix[i][j] > threshold:  # e.g., 0.5
            G.add_edge(df['Spotify ID'][i], df['Spotify ID'][j], weight=similarity_matrix[i][j])

node_color = []
for node in G.nodes():
    owner = df[df['Spotify ID'] == node]['PlaylistOwner'].values[0]
    node_color.append(color_map[owner])

hover_text = []
for node in G.nodes():
    song_info = df[df['Spotify ID'] == node]
    title = song_info['Track Name'].values[0]
    artist = song_info['Artist Name(s)'].values[0]
    hover_text.append(f'{title} <br>by {artist}')

pos = nx.spring_layout(G, dim=3)

edge_x, edge_y, edge_z = [], [], []
for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_z = [pos[node][2] for node in G.nodes()]

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z, 
    line=dict(width=0.5, color='#888'), 
    hoverinfo='none', 
    mode='lines',
    opacity=0.1,
    showlegend=False  # This will hide this trace from the legend
)

node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z, 
    mode='markers', 
    hoverinfo='text', 
    text=hover_text,
    marker=dict(size=5, color=node_color, line_width=1),
    showlegend=False  # This will hide this trace from the legend
)

traces = [edge_trace, node_trace] # edge_trace, 

for owner, color in color_map.items():
    traces.append(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup=owner,
        name=owner,
        showlegend=True
    ))

axis_layout = dict(showbackground=False, showline=True, zeroline=False, showgrid=True, showticklabels=False, title='')

layout = go.Layout(
    title='3D Network Graph of Spotify Songs',
    showlegend=True,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    scene=dict(
        xaxis=dict(axis_layout),
        yaxis=dict(axis_layout),
        zaxis=dict(axis_layout)
    ),
    paper_bgcolor='rgb(229,236,246)'
)


fig = go.Figure(data=traces, layout=layout)
fig.write_html('./../figs/network_graph-3d-songids.html')
fig.show()


# %%
