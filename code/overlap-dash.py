# %% 
# Import necessary libraries
import pandas as pd
import dash
import dash_cytoscape as cyto
from dash import html
from collections import defaultdict

# %% 
# Load the dataset
file_path = './../data/derived/combined.csv'  # Replace with your file path
data = pd.read_csv(file_path)
col = 'Track Name'

# %% 
# create graph

# Creating a dictionary to hold the artists for each owner
owner_artists = defaultdict(set)

# Populate the dictionary with data
for _, row in data.iterrows():
    owner_artists[row['PlaylistOwner']].add(row[col])

# Identify artists with overlap across different owners
artist_owners = defaultdict(set)

# Populate the artist_owners dictionary
for owner, artists in owner_artists.items():
    for artist in artists:
        artist_owners[artist].add(owner)

# Filter out artists that appear in only one owner's playlist
overlapping_artists = {artist: owners for artist, owners in artist_owners.items() if len(owners) > 1}

# Prepare elements for Cytoscape
elements = []
added_owners = set()  # To keep track of added owner nodes
for artist, owners in overlapping_artists.items():
    # Add node for each artist with a distinct style
    elements.append({
        'data': {'id': artist, 'label': artist},
        'style': {'background-color': '#0074D9', 'label': artist}
    })
    for owner in owners:
        # Add node for each playlist owner if not already added
        if owner not in added_owners:
            elements.append({
                'data': {'id': owner, 'label': owner},
                'style': {'background-color': '#FF4136', 'label': owner}
            })
            added_owners.add(owner)
        # Add edges between owners and artists
        elements.append({'data': {'source': owner, 'target': artist}})

# Initialize and run the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=elements,
        style={'width': '100%', 'height': '400px'},
        layout={
            'name': 'breadthfirst',
            'circle': True,
            'spacingFactor': 1.75  # Adjust spacing as needed
        }
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# %%

import plotly.graph_objects as go
import plotly.io as pio

# Manually create a Plotly figure (This is a simplified example)
fig = go.Figure()

# Add nodes and edges to the figure
for element in elements:
    if 'source' in element['data']:
        # Add edge
        # You'll need to determine the positions of source and target nodes
        fig.add_trace(go.Scatter(x=[source_x, target_x], y=[source_y, target_y], mode='lines'))
    else:
        # Add node
        # You'll need to determine the position of the node
        fig.add_trace(go.Scatter(x=[node_x], y=[node_y], mode='markers+text', text=element['data']['label']))

# Set layout properties (if needed)
fig.update_layout(showlegend=False)

# Save to HTML
pio.write_html(fig, file='cytoscape_graph.html')


# %%
