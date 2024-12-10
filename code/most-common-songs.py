#%%
import pandas as pd

# Load the dataset
file_path = './../data/derived/combined.csv'
data = pd.read_csv(file_path)

data = data[
    (data['Year'] == 2024) & 
    (data['PlaylistOwner'].str.lower().isin({'j', 'jason', 'braden', 'jon', 'jacob', 'theo', 'mcairth'}))
]

def find_top_tracks(data):
    """
    Find the 10 TrackIDs with the most entries and display their details,
    including which PlaylistOwner had the song for which years.

    :param data: DataFrame containing the song data.
    :return: DataFrame with Track Name, Number of Entries, and Playlist Owners with Years.
    """
    # Count the number of entries per TrackID
    top_tracks = data['TrackID'].value_counts().head(20)

    # Collect details for each top track
    records = []
    for track_id in top_tracks.index:
        track_rows = data[data['TrackID'] == track_id]
        track_name = track_rows.iloc[0]['Track Name']
        num_entries = top_tracks[track_id]
        
        owner_years_mapping = (
            track_rows.groupby('PlaylistOwner')['Year']
            .apply(lambda years: sorted(years.unique()))
            .to_dict()
        )

        records.append({
            # 'TrackID': track_id,
            'Track Name': track_name,
            'Number of Entries': num_entries,
            'Playlist Owners and Years': owner_years_mapping
        })

    return pd.DataFrame(records)

result = find_top_tracks(data)
result

# %%
