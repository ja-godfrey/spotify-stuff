import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import networkx as nx
import colorsys
from dash import Dash, dcc, html, Input, Output

# --- Data Loading and Processing ---

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

# Map track to PlaylistOwner(s)
owner_map = {}
for node in G.nodes():
    owners = df[df['TrackID'] == node]['PlaylistOwner'].unique()
    owner_map[node] = ', '.join(owners)

# Hover information
hover_text_map = {
    node: f"{df[df['TrackID'] == node]['Track Name'].values[0]}<br>"
          f"by {df[df['TrackID'] == node]['Artist Name(s)'].values[0]}<br>"
          f"PlaylistOwner(s): {owner_map[node]}"
    for node in G.nodes()
}

# Layout position
pos = nx.spring_layout(G, dim=3, seed=42)

# --- Function to Generate Plotly Graph ---

def create_figure(highlight_node=None):
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
    node_colors = [color_map[df[df['TrackID'] == node]['PlaylistOwner'].iloc[0]] for node in G.nodes()]
    
    # Muted colors for non-highlighted nodes
    if highlight_node:
        neighbors = set(G.neighbors(highlight_node))
        neighbors.add(highlight_node)
        node_colors = [
            color_map[df[df['TrackID'] == node]['PlaylistOwner'].iloc[0]] if node in neighbors else 'lightgrey' 
            for node in G.nodes()
        ]
        # Focus on the clicked node and its neighbors
        focus_x = [pos[node][0] for node in neighbors]
        focus_y = [pos[node][1] for node in neighbors]
        focus_z = [pos[node][2] for node in neighbors]
    else:
        focus_x = node_x
        focus_y = node_y
        focus_z = node_z

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='rgba(150,150,150,0.5)'),
        hoverinfo='none',
        mode='lines',
        opacity=0.2,
    )

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        text=[hover_text_map[node] for node in G.nodes()],
        marker=dict(size=6, color=node_colors, line=dict(width=0.5, color='black')),
    )

    layout = go.Layout(
        title='Music Similarity Network (2024)',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(
                    x=np.mean(focus_x),
                    y=np.mean(focus_y),
                    z=np.mean(focus_z) + 0.5  # Adjust height for a better view
                )
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest',
    )

    return go.Figure(data=[edge_trace, node_trace], layout=layout)

# --- Dash App ---

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='network-graph', figure=create_figure(), style={'height': '90vh'}),
    html.Div(id='node-info', style={'padding': '10px', 'fontSize': '16px'})
])

@app.callback(
    Output('network-graph', 'figure'),
    Input('network-graph', 'clickData')
)
def highlight_node(clickData):
    if clickData and 'points' in clickData and 'text' in clickData['points'][0]:
        node_text = clickData['points'][0]['text']
        for node, text in hover_text_map.items():
            if text == node_text:
                return create_figure(highlight_node=node)
    return create_figure()

if __name__ == '__main__':
    app.run_server(debug=True)
