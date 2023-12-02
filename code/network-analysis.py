# %%
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import random
import json
from tqdm import tqdm
import plotly.graph_objects as go
import json

#%%

# Directory containing the files
directory = './../data/raw/'  # Update this with your directory path

# Function to load and merge the playlists
def load_and_merge_playlists(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            person_name = filename.split("_")[0]
            df = pd.read_csv(os.path.join(directory, filename))
            df['PlaylistOwner'] = person_name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Load and merge the playlists
combined_df = load_and_merge_playlists(directory)

#%%
# create matplot graph just to see what it is like. save node positions to json.

# Step 1: Prepare the data - select song features for similarity calculation
features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
song_features = combined_df[features]

# Standardize the features
scaler = StandardScaler()
song_features_scaled = scaler.fit_transform(song_features)

# Step 2: Calculate similarity
similarity = cosine_similarity(song_features_scaled)

# Step 3: Construct the graph
G = nx.Graph()

# Adding nodes with the song title as the node label
for i, row in combined_df.iterrows():
    G.add_node(i, label=row['Track Name'], color=row['PlaylistOwner'])

# Add edges based on similarity
for i in range(len(similarity)):
    for j in range(i+1, len(similarity)):
        # Add an edge if similarity is above a certain threshold (e.g., 0.5)
        if similarity[i, j] > 0.5:
            G.add_edge(i, j, weight=similarity[i, j])

# Map user names to colors
unique_users = combined_df['PlaylistOwner'].unique()
colors = plt.cm.get_cmap('tab10', len(unique_users))  # Using a colormap with enough colors
user_color_map = {user: colors(i) for i, user in enumerate(unique_users)}

# Assign colors to nodes based on the user
color_map = [user_color_map[G.nodes[node]['color']] for node in G]

# Drawing the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, weight='weight')  # Positions nodes based on similarity

# Draw the nodes
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=50)

# Draw the edges with reduced opacity
nx.draw_networkx_edges(G, pos, alpha=0.1)

# Add a legend for colors
legend_labels = {user: plt.Line2D([0], [0], marker='o', color=color, label=user, markersize=10, linestyle='') 
                 for user, color in user_color_map.items()}
plt.legend(handles=legend_labels.values(), title="Users", bbox_to_anchor=(1.05, 1), loc='upper left')

# Randomly select nodes to label
labels = {}
for node in G.nodes():
    if random.randint(1, 50) == 1:  # Approximately 1% chance
        labels[node] = G.nodes[node]['label']

# Draw labels for the selected nodes
# Ensure they are drawn after nodes and edges
for node, label in labels.items():
    x, y = pos[node]
    plt.text(x, y, label, fontsize=8, color='red', fontweight='bold', ha='center', va='center')

plt.title("Song-Based Network Graph", size=15)
plt.savefig('./../figs/network.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
# create plotly graph

def extract_graph_data(G):
    """Extracts nodes and edges from a NetworkX graph."""
    nodes = [{'id': node, 'label': G.nodes[node]['label']} for node in G.nodes()]
    edges = [{'source': u, 'target': v} for u, v in G.edges()]
    return nodes, edges

def load_graph_data(filepath):
    """Loads graph data from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)

def create_contributor_color_map(df, column_name):
    """Creates a color map for contributors."""
    unique_contributors = df[column_name].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_contributors))
    return {contributor: f'rgb{colors(i)[:3]}' for i, contributor in enumerate(unique_contributors)}

def prepare_node_traces(graph_data, pos, df, contributor_color_map):
    """Prepares separate node traces for each contributor for the plot."""
    traces = []

    for contributor, color in contributor_color_map.items():
        x, y, text = [], [], []
        for node in graph_data['nodes']:
            if df.loc[node['id'], 'PlaylistOwner'] == contributor:
                x_pos, y_pos = pos[node['id']]
                x.append(x_pos)
                y.append(y_pos)
                hover_text = (
                    f"Song: {node['label']}<br>"
                    f"Artist: {df.loc[node['id'], 'Artist Name(s)']}<br>"
                    f"Genre: {df.loc[node['id'], 'Genres']}"
                )
                text.append(hover_text)

        trace = go.Scatter(
            x=x, y=y,
            text=text,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                size=10,
                opacity=0.8,
                color=color
            ),
            legendgroup=contributor,
            name=contributor,
            showlegend=True
        )
        traces.append(trace)

    return traces

def create_figure(traces):
    """Creates the plotly figure with the given node trace and legend."""
    layout=go.Layout(
        title='Song Network',
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        annotations=[dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

# Main Execution
G = nx.Graph()  # Assuming G is a predefined NetworkX graph
combined_df = combined_df  # Assuming combined_df is a predefined DataFrame
pos = pos  # Assuming pos is a predefined dictionary for node positions

nodes, edges = extract_graph_data(G)
graph_data = load_graph_data('./../data/derived/network_graph_data.json')
contributor_color_map = create_contributor_color_map(combined_df, 'PlaylistOwner')
traces = prepare_node_traces(graph_data, pos, combined_df, contributor_color_map)
fig = create_figure(traces)
fig.show()


# %%
