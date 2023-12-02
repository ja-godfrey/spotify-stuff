# %%
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
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

# Create a new dataframe for user similarity
user_artist_df = combined_df.groupby('PlaylistOwner')['Artist Name(s)'].apply(lambda x: "%s" % ' '.join(x)).reset_index()

# Vectorize the artist names to create a user-artist matrix
vectorizer = CountVectorizer()
user_artist_matrix = vectorizer.fit_transform(user_artist_df['Artist Name(s)'])

# Compute the cosine similarity between users
cosine_sim = cosine_similarity(user_artist_matrix)

# Create a DataFrame from the cosine similarity matrix
similarity_df = pd.DataFrame(cosine_sim, index=user_artist_df['PlaylistOwner'], columns=user_artist_df['PlaylistOwner'])

# Create a network graph
G = nx.Graph()

# Add nodes and edges with weights
for user in similarity_df.index:
    for other_user in similarity_df.columns:
        if user != other_user and similarity_df.loc[user, other_user] > 0.2:
            G.add_edge(user, other_user, weight=similarity_df.loc[user, other_user])

# Draw the network graph
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("User Similarity Network Based on Spotify Playlists")
plt.show()

# %%

