#%%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the dataset
file_path = './../data/derived/combined.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Creating a graph
G = nx.Graph()

# Creating a dictionary to hold the artists for each owner
owner_artists = defaultdict(set)

# Populating the dictionary with data
for _, row in data.iterrows():
    owner_artists[row['PlaylistOwner']].add(row['Artist Name(s)'])

# Identifying artists with overlap across different owners
# We will find artists who appear in the playlists of more than one owner

# Dictionary to keep track of the number of owners per artist
artist_owners = defaultdict(set)

# Populating the dictionary
for owner, artists in owner_artists.items():
    for artist in artists:
        artist_owners[artist].add(owner)

# Filtering out artists that appear in only one owner's playlist
overlapping_artists = {artist: owners for artist, owners in artist_owners.items() if len(owners) > 1}

# Creating a new graph for only overlapping artists
G_overlap = nx.Graph()

# Adding nodes and edges for overlapping artists
for artist, owners in overlapping_artists.items():
    for owner in owners:
        G_overlap.add_edge(owner, artist)

# Drawing the new graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G_overlap, k=0.15, iterations=20)
nx.draw_networkx_nodes(G_overlap, pos, node_size=50, node_color='blue')
nx.draw_networkx_edges(G_overlap, pos, alpha=0.5)
plt.title("Network Graph of Playlist Owners and Overlapping Artists")
plt.show()

# %%
