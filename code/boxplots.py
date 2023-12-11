#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = './../data/derived/a-combined.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Relevant song metrics
metrics = ['Popularity', 'Danceability', 'Energy', 'Loudness', 'Speechiness', 
           'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']


# Ensure metrics are numeric
data[metrics] = data[metrics].apply(pd.to_numeric, errors='coerce')

# Melt the dataframe to have a long format for seaborn
melted_data = pd.melt(data, id_vars=['PlaylistOwner'], value_vars=metrics, 
                      var_name='Metric', value_name='Value')

# Get unique PlaylistOwners
playlist_owners = melted_data['PlaylistOwner'].unique()

# Create a color palette (choose a palette or define custom colors)
palette = sns.color_palette("deep", len(playlist_owners))
color_dict = dict(zip(playlist_owners, palette))

# Create a separate plot for each metric
for metric in metrics:
    metric_data = melted_data[melted_data['Metric'] == metric]

    # Determine the order of PlaylistOwner based on the median value of the metric
    order = metric_data.groupby('PlaylistOwner')['Value'].median().dropna().sort_values(ascending=False).index

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Value', y='PlaylistOwner', data=metric_data, order=order, palette=color_dict)
    
    plt.title(f'Distribution of {metric} Across Playlist Owners')
    plt.xlabel('Metric Value')
    plt.ylabel('Playlist Owner')
    
    plt.show()

# %%
