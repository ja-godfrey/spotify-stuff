# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import pi

# Load the dataset
file_path = './../data/derived/combined.csv'
df = pd.read_csv(file_path)

df = df[
    (df['Year'] == 2024) & 
    (df['PlaylistOwner'].str.lower().isin({'j', 'jason', 'braden', 'jon', 'jacob', 'theo', 'mcairth'}))
]

# Processing the 'Release Date' to extract the year and convert it to numeric
df['Release Year'] = pd.to_datetime(df['Release Date'], errors='coerce').dt.year

# Selecting relevant columns for the analysis
columns_to_analyze = [
    'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 
    'Liveness', 'Valence', 'Tempo', 'Danceability', 'Energy', 'Popularity', 
    'Release Year', 'Duration (ms)'
]
df[columns_to_analyze] = df[columns_to_analyze].replace(0, pd.NA)
df.dropna(subset=columns_to_analyze, inplace=True)

# Group by 'PlaylistOwner' and calculate the mean for each column
grouped_df = df.groupby('PlaylistOwner')[columns_to_analyze].mean()

# Normalizing the data for better visualization in radar chart
scaler = MinMaxScaler()
grouped_df_scaled = pd.DataFrame(scaler.fit_transform(grouped_df), index=grouped_df.index, columns=grouped_df.columns)

# Function to create a radar chart for each playlist owner
def create_radar_chart(data, owner):
    categories = list(data.index)
    N = len(categories)

    # We need to repeat the first value to close the circular graph:
    values = data.values.flatten().tolist()
    values += values[:1]
    categories += categories[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], data.index, color='grey', size=8)
    plt.ylim(-0.2, 1)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=owner)

    # Fill area
    ax.fill(angles, values, alpha=0.1)

    plt.title(owner, size=11, color='blue', y=1.1)

# Generate a radar chart for each playlist owner
for owner in grouped_df_scaled.index:
    plt.figure(figsize=(6, 6))
    create_radar_chart(grouped_df_scaled.loc[owner], owner)
    plt.show()

#%%
