# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Load and filter data
file_path = './../data/derived/combined.csv'  # Replace with your file path
data = pd.read_csv(file_path)
data = data[(data['Year'] == 2024) & 
            (data['PlaylistOwner'].str.lower().isin({'tuquyen', 'jason', 'jen', 'narric', 'nakia', 'kevin', 'milagro', 'kim'}))]

# Preprocess data
# Combine relevant features to create a rich text representation for each user
user_artist_df = data.groupby('PlaylistOwner').apply(
    lambda x: " ".join(x['Artist Name(s)'] + " " + x['Genres'].fillna('') + " " + x['Record Label'].fillna(''))
).reset_index(name='CombinedFeatures')

# Normalize numeric features and add to the user feature matrix
numeric_cols = ['Popularity', 'Danceability', 'Energy', 'Valence', 'Tempo']
scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols].fillna(0))
user_numeric_features = data.groupby('PlaylistOwner')[numeric_cols].mean()

# Use TF-IDF Vectorizer for text-based features
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(user_artist_df['CombinedFeatures'])

# Combine text-based and numeric features
combined_features = np.hstack((text_features.toarray(), user_numeric_features.values))

# Compute cosine similarity between users
cosine_sim = cosine_similarity(combined_features)

# Create a DataFrame from the similarity matrix
similarity_df = pd.DataFrame(cosine_sim, index=user_artist_df['PlaylistOwner'], columns=user_artist_df['PlaylistOwner'])

# Create a network graph
G = nx.Graph()

# Add nodes and edges with weights
for user in similarity_df.index:
    G.add_node(user, size=user_numeric_features.loc[user, 'Popularity'] * 100)
    for other_user in similarity_df.columns:
        if user != other_user and similarity_df.loc[user, other_user] > 0.3:
            G.add_edge(user, other_user, weight=similarity_df.loc[user, other_user])

# Draw the network graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)

# Node sizes based on average popularity
node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# Edge widths based on similarity scores
edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.7)

# Add edge labels with similarity scores rounded to 2 decimal places
edge_labels = nx.get_edge_attributes(G, 'weight')
edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("User Similarity Network Based on Spotify Playlists", fontsize=18)
plt.savefig("./../figs/2024/accelerate/user_similarity_network.png", format="png", dpi=300)
plt.show()
# %%
# Heatmap of User Similarity
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True,
            xticklabels=similarity_df.columns, yticklabels=similarity_df.index)
plt.title("Heatmap of User Similarity", fontsize=16)
plt.xlabel("Playlist Owners", fontsize=12)
plt.ylabel("Playlist Owners", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("./../figs/2024/accelerate/user_similarity_heatmap.png", format="png", dpi=300)
plt.show()
# %%