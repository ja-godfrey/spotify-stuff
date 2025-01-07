# %%
# Import necessary libraries
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
import math
import numpy as np

# Load data
file_path = './../data/derived/combined.csv'  # Replace with your file path
data = pd.read_csv(file_path)

data = pd.read_csv('./../data/derived/combined.csv')
data = data[
    (data['Year'] == 2024) & 
    # (data['PlaylistOwner'].str.lower().isin({'j', 'jason', 'braden', 'jon', 'jacob', 'theo', 'mcairth'}))
    (data['PlaylistOwner'].str.lower().isin({'tuquyen', 'jason', 'jen', 'narric', 'narric', 'nakia', 'kevin', 'milagro', 'kim'}))
]
col = 'Artist Name(s)'

# Prepare data for the network graph
owner_artists = defaultdict(set)
artist_owners = defaultdict(set)

# Populate the dictionaries with data
for _, row in data.iterrows():
    owner_artists[row['PlaylistOwner']].add(row[col])
    artist_owners[row[col]].add(row['PlaylistOwner'])

# Filter out artists that appear in only one owner's playlist
overlapping_artists = {artist: owners for artist, owners in artist_owners.items() if len(owners) > 1}

# Create the network graph
G = nx.Graph()

# Add nodes and edges for overlapping artists
for artist, owners in overlapping_artists.items():
    for owner in owners:
        G.add_edge(owner, artist)

# Generate positions for the nodes in the graph using a breadth-first layout
def bfs_layout_circular(G, root=None):
    if root is None:
        root = list(G.nodes)[0]

    levels = {}
    levels[root] = 0
    queue = [root]
    while queue:
        node = queue.pop(0)
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            if neighbor not in levels:
                levels[neighbor] = levels[node] + 1
                queue.append(neighbor)

    max_level = max(levels.values())
    pos = {}
    for level in range(max_level + 1):
        nodes_at_level = [node for node, lvl in levels.items() if lvl == level]
        num_nodes = len(nodes_at_level)
        angle = 2 * math.pi / num_nodes
        for i, node in enumerate(nodes_at_level):
            theta = i * angle
            pos[node] = (0.5 + math.cos(theta) / (level + 1), 0.5 + math.sin(theta) / (level + 1))

    return pos

# Use the bfs_layout function
pos = bfs_layout_circular(G)

# Cell 6: Create Plotly trace for the edges
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# Cell 7: Create Plotly trace for the nodes
node_x = []
node_y = []
node_info = {}
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_info = {}
for _, row in data.iterrows():
    artist = row[col]  # Assuming 'col' holds the artist name
    owner = row['PlaylistOwner']
    track_name = row['Track Name']
    artist_name = row['Artist Name(s)']  # Replace with the actual column name for artist names in your data
    node_info[artist] = (track_name, artist_name)
    node_info[owner] = (track_name, artist_name)

# Cell 8: Color node points by the number of connections
node_adjacencies = []
# Modified Step: Update Node Text for Hover Information
node_color = []

node_text = []
for node in G.nodes():
    adjacencies = list(G.adj[node])
    num_connections = len(adjacencies)

    # Check if the node is a user or an artist
    if node in owner_artists:
        # User node - Show only user name and number of connections
        node_color.append('#FF6961')  # Or your chosen color for user nodes
        hover_text = f"User: {node}<br># of connections: {num_connections}"
    else:
        # Artist node - Show detailed information including connections
        node_color.append('#61A8FF')  # Or your chosen color for artist nodes
        connections = ', '.join(adjacencies)
        track_name, artist_name = node_info.get(node, ("Unknown", "Unknown"))
        hover_text = f"Track: {track_name}<br>Artist: {artist_name}<br># of connections: {num_connections}<br>Connections: {connections}"

    node_text.append(hover_text)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        size=10,
        color=node_color,  # Use the predefined node_color list
        line_width=2))

node_trace.marker.color = node_color
node_trace.text = node_text

# Cell 9: Create the network graph
fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(
                    text='Overlapping Tracks',
                    x=0.5,  # Centers the title
                    xanchor='center',  # Ensures the title is centered at the given x position
                    font=dict(size=24)
                ),
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                paper_bgcolor='rgb(229,236,246)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

# Cell 10: Show the Plotly graph
fig.write_html('./../figs/overlapping_tracks.html')
fig.show()


# %%
