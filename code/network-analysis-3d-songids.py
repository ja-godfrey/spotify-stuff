# %%
# https://chat.openai.com/c/9f85b511-eb36-4216-bde6-29eb2d1a9be5 2023
# https://chatgpt.com/c/675532a1-bd4c-8002-8335-e6e7e0dc3016?model=gpt-4o 2024
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import networkx as nx
import colorsys

threshold = 0.7
df = pd.read_csv('./../data/derived/combined.csv')
df = df[
    (df['Year'] == 2024) & 
    (df['PlaylistOwner'].str.lower().isin({'j', 'jason', 'braden', 'jon', 'jacob', 'theo', 'mcairth'}))
]

columns_to_analyze = [
    'Duration (ms)', 'Popularity', 'Danceability', 'Energy', 'Key', 'Loudness','Speechiness', 'Acousticness', 
    'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Time Signature'
]
df_unique = df.drop_duplicates(subset='TrackID')[columns_to_analyze].reset_index(drop=True)
track_ids = df.drop_duplicates(subset='TrackID')['TrackID'].reset_index(drop=True)

def generate_pastel_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
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
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i][j] > threshold:
            G.add_edge(track_ids[i], track_ids[j], weight=similarity_matrix[i][j])

owner_map = {node: df[df['TrackID'] == node]['PlaylistOwner'].values[0] for node in G.nodes()}
hover_text = [
    f"{df[df['TrackID'] == node]['Track Name'].values[0]}<br>by {df[df['TrackID'] == node]['Artist Name(s)'].values[0]}"
    for node in G.nodes()
]

dissimilarity = 1 - similarity_matrix
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
embeddings = mds.fit_transform(dissimilarity)
pos = {track_ids[i]: embeddings[i] for i in range(len(track_ids)) if track_ids[i] in G.nodes()}

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
node_colors = [color_map[owner_map[node]] for node in G.nodes()]

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z, 
    line=dict(width=0.5, color='rgba(150,150,150,0.25)'), 
    hoverinfo='none', 
    mode='lines',
    opacity=0.2,
    showlegend=False
)

node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z, 
    mode='markers', 
    hoverinfo='text', 
    text=hover_text,
    marker=dict(size=6, color=node_colors, line=dict(width=0.5, color='black')),
    showlegend=False
)

traces = [edge_trace, node_trace]
for owner, color in color_map.items():
    traces.append(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        legendgroup=owner,
        name=owner,
        showlegend=True
    ))

layout = go.Layout(
    title='Music Similarity Network (2024)',
    showlegend=True,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    paper_bgcolor='rgb(245,245,245)',
    scene=dict(
        xaxis=dict(showbackground=True, backgroundcolor='rgb(235,235,235)', showgrid=False, zeroline=False),
        yaxis=dict(showbackground=True, backgroundcolor='rgb(235,235,235)', showgrid=False, zeroline=False),
        zaxis=dict(showbackground=True, backgroundcolor='rgb(235,235,235)', showgrid=False, zeroline=False)
    )
)

fig = go.Figure(data=traces, layout=layout)
fig.write_html('./../figs/network_graph-3d-improved.html')
fig.show()
# %%
# Export
edges_data = [
    {'Source': edge[0], 'Target': edge[1], 'Similarity': G[edge[0]][edge[1]]['weight']}
    for edge in G.edges()
]

# Convert to DataFrame
edges_df = pd.DataFrame(edges_data)

# Export to CSV
edges_df.to_csv('./../data/derived/relationships.csv', index=False)

# Export to JSON
edges_df.to_json('./../data/derived/relationships.json', orient='records', indent=4)

# %%

