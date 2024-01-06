#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = './../data/derived/combined.csv'  # Replace with your file path
data = pd.read_csv(file_path)
data = data[~data['PlaylistOwner'].str.contains('robert', na=False)]

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
    
    plt.savefig(f'./../figs/bros/boxplot_{metric}.png', bbox_inches='tight', dpi=300)
    plt.show()

# %%

def find_extreme_songs_by_metric(data, metric):
    """
    Find the songs with the highest and lowest values for a given metric.

    :param data: DataFrame containing the song data.
    :param metric: The metric to evaluate (e.g., 'Popularity', 'Loudness').
    :return: Information about the songs with the highest and lowest values for the metric, including the contributor.
    """
    if metric not in data.columns:
        return "Invalid metric. Please choose a valid metric from the dataset."

    # Find the rows with the highest and lowest values for the metric
    max_value_row = data.loc[data[metric].idxmax()]
    min_value_row = data.loc[data[metric].idxmin()]

    # Extracting information including the contributor
    max_info = {
        'Song': max_value_row['Track Name'],
        'Artist': max_value_row['Artist Name(s)'],
        'Album': max_value_row['Album Name'],
        'Metric Value': max_value_row[metric],
        'Contributor': max_value_row['PlaylistOwner']
    }

    min_info = {
        'Song': min_value_row['Track Name'],
        'Artist': min_value_row['Artist Name(s)'],
        'Album': min_value_row['Album Name'],
        'Metric Value': min_value_row[metric],
        'Contributor': min_value_row['PlaylistOwner']
    }

    return max_info, min_info


# metrics = ['Popularity', 'Danceability', 'Energy', 'Loudness', 'Speechiness', 
#            'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']

metric = 'Danceability'  # Replace with the metric you want to check
highest, lowest = find_extreme_songs_by_metric(data, metric)
print(f"Highest: {highest} \nLowest:, {lowest}")

# %%

metrics = ['Popularity', 'Danceability', 'Energy', 'Loudness', 'Speechiness', 
           'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']

# Open a file to write the results
with open('./../figs/extreme_songs_by_metrics.txt', 'w', encoding='utf-8') as file:
    for m in metrics:
        highest, lowest = find_extreme_songs_by_metric(data, m)
        file.write(f"Metric: {m}\n")
        file.write("Highest: \n")
        for key, value in highest.items():
            file.write(f"\t{key}: {value}\n")
        file.write("Lowest: \n")
        for key, value in lowest.items():
            file.write(f"\t{key}: {value}\n")
        file.write("\n")
# %%
