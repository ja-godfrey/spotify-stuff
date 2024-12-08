#%%
import pandas as pd
import os
import re

directory = './../data/raw/'

# Function to load and merge all CSV files in the directory and subdirectories
def load_and_merge_playlists(directory):
    dfs = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv"):
                # Regular expression to find a 4-digit year
                match = re.search(r'(\d{4})', filename)
                year = int(match.group(1)) if match else None

                # Extract the person's name by removing the year and '.csv' extension
                person_name = re.sub(r'\d{4}', '', filename).replace('.csv', '').strip('_')

                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path)

                # Standardize 'Spotify ID' and 'Track ID' into a single 'TrackID' column
                if 'Spotify ID' in df.columns:
                    df.rename(columns={'Spotify ID': 'TrackID'}, inplace=True)
                elif 'Track ID' in df.columns:
                    df.rename(columns={'Track ID': 'TrackID'}, inplace=True)

                df['PlaylistOwner'] = person_name
                df['Year'] = year
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

combined_df = load_and_merge_playlists(directory)

combined_df.to_csv('./../data/derived/combined.csv', index=False)

# %%
# Who is missing
names_to_check = {'j', 'jason', 'braden', 'jon', 'jacob', 'theo', 'mcairth'}

# Filter years where any of the names are missing
missing_names_by_year = combined_df.groupby('Year')['PlaylistOwner'].apply(
    lambda owners: names_to_check - set(map(str.lower, owners))
)

# Get years where any names are missing
years_with_missing_names = missing_names_by_year[missing_names_by_year.apply(bool)]

# Print years and the missing names
print(years_with_missing_names)
# %%
