# %%
import pandas as pd
import numpy as np


file_path = './../data/derived/combined.csv' 
data = pd.read_csv(file_path)

user_groups = {
    "Accelerate":{'tuquyen', 'jason', 'jen', 'narric', 'nakia', 'kevin', 'milagro', 'kim'},
    "bros": ["braden", "j", "jacob", "theo", "mcairth", "jon"],
    "fam": ["christina", "tiana", "jacob", "daniel", "liam"]
}

import plotly.graph_objects as go

# Create histogram data
hist_data = data['Duration (ms)'].values

# Create histogram figure
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=hist_data,
    nbinsx=50,
    name='Duration Distribution',
    marker_color='#4C78A8'
))

# Update layout
fig.update_layout(
    title='Distribution of Song Durations',
    xaxis_title='Duration (ms)',
    yaxis_title='Count',
    bargap=0.1,
    template='plotly_white'
)

# Show plot
fig.show()








# %%
