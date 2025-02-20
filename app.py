import base64
import io

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')


# ---------- Existing NLTK-based code starts here ----------

def create_semantic_graph(sentence):
    """Creates a semantic graph with weighted relationships."""
    WEIGHTS = {
        'exact': 1.0,
        'synonym': 0.9,
        'hypernym': 0.8,
        'hyponym': 0.8,
        'meronym': 0.6,
        'holonym': 0.6
    }

    G = nx.DiGraph()
    words = nltk.word_tokenize(sentence.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    tagged_words = nltk.pos_tag(filtered_words)

    # Track all nodes and their relationships
    word_synsets = {}
    # Track node types for visualization
    node_types = {}

    for word, pos in tagged_words:
        if pos[0] in ['N', 'V', 'J']:  # Consider nouns, verbs, and adjectives
            word_node = f"{word}_{pos}"
            G.add_node(word_node, type='word')
            node_types[word_node] = 'word'

            synsets = wn.synsets(word)
            if not synsets:
                continue

            for synset in synsets:
                synset_node = str(synset.name())
                G.add_node(synset_node, type='synset')
                node_types[synset_node] = 'synset'
                G.add_edge(word_node, synset_node, weight=WEIGHTS['exact'])

                if word_node not in word_synsets:
                    word_synsets[word_node] = []
                word_synsets[word_node].append(synset)

                # Hypernyms
                for hypernym in synset.hypernyms():
                    hyp_node = str(hypernym.name())
                    G.add_node(hyp_node, type='hypernym')
                    node_types[hyp_node] = 'hypernym'
                    G.add_edge(synset_node, hyp_node, weight=WEIGHTS['hypernym'])

                # Hyponyms
                for hyponym in synset.hyponyms():
                    hypo_node = str(hyponym.name())
                    G.add_node(hypo_node, type='hyponym')
                    node_types[hypo_node] = 'hyponym'
                    G.add_edge(synset_node, hypo_node, weight=WEIGHTS['hyponym'])

                # Meronyms
                for meronym in synset.part_meronyms() + synset.member_meronyms():
                    mero_node = str(meronym.name())
                    G.add_node(mero_node, type='meronym')
                    node_types[mero_node] = 'meronym'
                    G.add_edge(synset_node, mero_node, weight=WEIGHTS['meronym'])

                # Holonyms
                for holonym in synset.part_holonyms() + synset.member_holonyms():
                    holo_node = str(holonym.name())
                    G.add_node(holo_node, type='holonym')
                    node_types[holo_node] = 'holonym'
                    G.add_edge(synset_node, holo_node, weight=WEIGHTS['holonym'])

    return G, word_synsets, node_types

def calculate_word_similarity(synsets1, synsets2):
    """Calculate similarity between two sets of synsets."""
    max_sim = 0.0
    for syn1 in synsets1:
        for syn2 in synsets2:
            # Check for exact match or synonym
            if syn1 == syn2:
                return 1.0
            lemmas1 = set(l.name() for l in syn1.lemmas())
            lemmas2 = set(l.name() for l in syn2.lemmas())
            if lemmas1.intersection(lemmas2):
                return 0.9

            # Calculate path similarity
            path_sim = syn1.path_similarity(syn2) or 0.0

            # Check hypernym/hyponym relationship
            if syn2 in syn1.hypernyms() or syn1 in syn2.hypernyms():
                path_sim = max(path_sim, 0.8)

            max_sim = max(max_sim, path_sim)
    return max_sim

def visualize_semantic_graph(G, node_types, title, ax=None):
    """Visualize the semantic graph with different colors for node types."""
    if ax is None:
        ax = plt.gca()

    color_map = {
        'word': '#FF9999',      # Light red
        'synset': '#99FF99',    # Light green
        'hypernym': '#9999FF',  # Light blue
        'hyponym': '#FFFF99',   # Light yellow
        'meronym': '#FF99FF',   # Light purple
        'holonym': '#99FFFF'    # Light cyan
    }

    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw nodes by type
    for node_type in color_map:
        node_list = [node for node, t in node_types.items() if t == node_type]
        if node_list:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=node_list,
                node_color=color_map[node_type],
                node_size=2000,
                alpha=0.6,
                ax=ax
            )

    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax)

    labels = {
        node: node.split('.')[0] if '.' in node else node.split('_')[0]
        for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis('off')

def compare_sentences_and_return_image(sentence1, sentence2):
    """
    Compare two sentences using semantic similarity and return:
      - The similarity score
      - A base64-encoded PNG image of the two semantic graphs
    """
    # Create semantic graphs
    graph1, words1, types1 = create_semantic_graph(sentence1)
    graph2, words2, types2 = create_semantic_graph(sentence2)

    # Calculate similarities
    similarities = []
    for synset_list1 in words1.values():
        word_sims = []
        for synset_list2 in words2.values():
            sim = calculate_word_similarity(synset_list1, synset_list2)
            word_sims.append(sim)
        if word_sims:
            similarities.append(max(word_sims))

    if not similarities:
        final_score = 0.0
    else:
        avg_similarity = sum(similarities) / len(similarities)
        coverage = len(similarities) / max(len(words1), len(words2))

        if coverage > 0.8 and avg_similarity > 0.7:
            final_score = min(1.0, avg_similarity * 1.2)
        else:
            final_score = avg_similarity

    # Plot the graphs side by side
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    visualize_semantic_graph(graph1, types1, f"Semantic Graph: {sentence1}", ax1)
    visualize_semantic_graph(graph2, types2, f"Semantic Graph: {sentence2}", ax2)

    # Convert the figure to a base64-encoded PNG
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return final_score * 100, encoded_image

# ---------- Dash Application (Dark Theme) ----------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Semantic Diff Checker"

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("Semantic Diff Checker", className="text-center my-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Text 1"),
                        dcc.Textarea(
                            id="sentence1-input",
                            style={"width": "100%", "height": "200px"},
                            placeholder="Enter first text..."
                        ),
                    ],
                    md=6
                ),
                dbc.Col(
                    [
                        html.Label("Text 2"),
                        dcc.Textarea(
                            id="sentence2-input",
                            style={"width": "100%", "height": "200px"},
                            placeholder="Enter second text..."
                        ),
                    ],
                    md=6
                ),
            ],
            className="mb-4"
        ),
        dbc.Button("Compare", id="compare-button", color="primary", className="mb-3"),
        html.Div(id="similarity-output", className="h4 mb-4"),
        html.Div(
            id="graph-output",
            style={"textAlign": "center"}
        )
    ],
    style={"backgroundColor": "#2b2b2b", "color": "white", "padding": "20px"}
)

@app.callback(
    [Output("similarity-output", "children"),
     Output("graph-output", "children")],
    [Input("compare-button", "n_clicks")],
    [State("sentence1-input", "value"), State("sentence2-input", "value")]
)
def compare_texts(n_clicks, sentence1, sentence2):
    if not n_clicks:
        return "", ""

    if not sentence1 or not sentence2:
        return "Please enter text in both fields.", ""

    similarity_score, encoded_image = compare_sentences_and_return_image(sentence1, sentence2)

    similarity_text = f"Similarity Score: {similarity_score:.2f}%"

    # Create an <img> tag with the base64-encoded PNG
    image_element = html.Img(
        src=f"data:image/png;base64,{encoded_image}",
        style={"maxWidth": "100%", "border": "2px solid white", "marginTop": "20px"}
    )

    return similarity_text, image_element

if __name__ == "__main__":
    app.run_server(debug=True)
