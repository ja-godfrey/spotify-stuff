#%%
import pandas as pd

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

combined_df.to_csv('./../data/derived/combined.csv', index=False)
# %%
