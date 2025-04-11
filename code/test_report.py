import plotly.express as px

# --- Test chart ---

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
import math

file_path = './../data/derived/combined.csv'
data = pd.read_csv(file_path)
data = data[
    (data['Year'] == 2024) & 
    (data['PlaylistOwner'].str.lower().isin({'tuquyen', 'jason', 'jen', 'narric', 'nakia', 'kevin', 'milagro', 'kim'}))
]

col = 'Artist Name(s)'

owner_artists = defaultdict(set)
artist_owners = defaultdict(set)

for _, row in data.iterrows():
    owner_artists[row['PlaylistOwner']].add(row[col])
    artist_owners[row[col]].add(row['PlaylistOwner'])

overlapping_artists = {artist: owners for artist, owners in artist_owners.items() if len(owners) > 1}

G = nx.Graph()
for artist, owners in overlapping_artists.items():
    for owner in owners:
        G.add_edge(owner, artist)

def bfs_layout_circular(G, root=None):
    if root is None:
        root = list(G.nodes)[0]
    levels = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbor in G.neighbors(node):
            if neighbor not in levels:
                levels[neighbor] = levels[node] + 1
                queue.append(neighbor)
    max_level = max(levels.values())
    pos = {}
    for level in range(max_level + 1):
        nodes_at_level = [n for n, lvl in levels.items() if lvl == level]
        angle = 2 * math.pi / len(nodes_at_level)
        for i, n in enumerate(nodes_at_level):
            theta = i * angle
            pos[n] = (0.5 + math.cos(theta) / (level + 1), 0.5 + math.sin(theta) / (level + 1))
    return pos

pos = bfs_layout_circular(G)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#aaa'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_color = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    adjacencies = list(G.adj[node])
    num_connections = len(adjacencies)
    if node in owner_artists:
        node_color.append('#fa8072')
        hover_text = f"<b>User:</b> {node}<br><b>Connections:</b> {num_connections}"
    else:
        node_color.append('#87ceeb')
        connections = ', '.join(adjacencies)
        hover_text = f"<b>Artist:</b> {node}<br><b>Connections:</b> {num_connections}<br><b>Connected To:</b> {connections}"
    node_text.append(hover_text)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    textposition='top center',
    marker=dict(size=15, color=node_color, line_width=2),
    textfont=dict(color='black', size=10),
    text=[n if n in owner_artists else '' for n in G.nodes()]
)

node_trace.text = [n if n in owner_artists else '' for n in G.nodes()]
node_trace.hovertext = node_text

network_fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text='Overlapping Artists Network',
                        x=0.5,
                        xanchor='center',
                        font=dict(size=24)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    dragmode='pan'
                ))

network_fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        direction="left",
        buttons=[
            dict(label="Reset View",
                 method="relayout",
                 args=[{"xaxis.range": [min(node_x)-0.1, max(node_x)+0.1],
                        "yaxis.range": [min(node_y)-0.1, max(node_y)+0.1]}])
        ],
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.05,
        xanchor="left",
        y=1.1,
        yanchor="top"
    )]
)

network_fig.write_html('./../figs/2024/accelerate/overlapping_artists_improved.html')
network_fig.show()

# --- Who is this report for? ---
audience = "fam"

# --- Create charts with dark template ---
chart1 = px.bar(x=["A", "B", "C", "D"], y=[10, 22, 28, 43], title="Chart 1: Basic Bar", template="plotly_dark")
chart2 = px.line(x=["Q1", "Q2", "Q3", "Q4"], y=[15, 30, 45, 60], title="Chart 2: Line Chart", template="plotly_dark")
chart3 = px.pie(names=["Group A", "Group B", "Group C"], values=[40, 25, 35], title="Chart 3: Pie Chart", template="plotly_dark")

# --- Assemble HTML parts ---
html_parts = [
    "<section>",
    "<h1>📊 Annual Music Note</h1>",
    "<p class='lead'>2024 Wrapped</p>",
    "</section>",

    "<section>",
    "<h2>Section 1: The Network</h2>",
    "<p>This chart shows the connections between users and artists in the dataset. Users are represented by red nodes, while artists are represented by blue nodes. The connections show which users have the same artists in their playlists.</p>",
    network_fig.to_html(full_html=False, include_plotlyjs=False),
    "</section>",

    "<section>",
    "<h3>Subsection 1.1: Trends</h3>",
    "<p>Praesent sagittis eros a massa fermentum, eget tincidunt justo efficitur.</p>",
    "</section>",

    "<section>",
    "<h2>Section 2: Detailed Findings</h2>",
    "<p>Integer posuere erat a ante venenatis dapibus posuere velit aliquet.</p>",
    chart2.to_html(full_html=False, include_plotlyjs=False),
    "</section>",

    "<section>",
    "<h3>Subsection 2.1: Segment Analysis</h3>",
    "<p>Nulla vitae elit libero, a pharetra augue. Morbi leo risus, porta ac consectetur ac, vestibulum at eros.</p>",
    chart3.to_html(full_html=False, include_plotlyjs=False),
    "</section>"
]

# --- Final HTML wrapper with dark theme ---
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Data Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            max-width: 880px;
            margin: 0 auto;
            padding: 4em 2em;
            background-color: #111;
            color: #eee;
        }}
        h1 {{
            font-size: 2.8em;
            margin-bottom: 0.2em;
            color: #fff;
        }}
        h2 {{
            font-size: 1.8em;
            margin-top: 3em;
            border-bottom: 2px solid #444;
            padding-bottom: 0.3em;
            color: #ddd;
        }}
        h3 {{
            font-size: 1.4em;
            margin-top: 2em;
            color: #ccc;
        }}
        p {{
            font-size: 1.05em;
            margin-top: 1em;
            line-height: 1.7;
            color: #bbb;
        }}
        .lead {{
            font-size: 1.2em;
            color: #aaa;
            margin-bottom: 2em;
        }}
        section {{
            background: #1b1b1b;
            padding: 2em;
            margin: 2em 0;
            border-radius: 12px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.6);
        }}
        .plotly-graph-div {{
            margin-top: 2em;
        }}
    </style>
</head>
<body>
    {''.join(html_parts)}
</body>
</html>
"""

# --- Write to file with UTF-8 encoding ---
with open("report.html", "w", encoding="utf-8") as f:
    f.write(html_template)
