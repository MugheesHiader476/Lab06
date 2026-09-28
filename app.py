import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from algorithms import (
    build_graph, coordinates, compute_heuristics, gbfs, astar
)

st.set_page_config(page_title="Search Algorithms Visualizer",
                   layout="wide")
st.title("🔍 GBFS vs A* — Airport Baggage Handling")

G = build_graph()
node_list = list(coordinates.keys())

st.sidebar.header("Configuration")
start = st.sidebar.selectbox("Start Node", node_list, index=0)
goal = st.sidebar.selectbox("Goal Node", node_list, index=len(node_list) - 1)
algorithm = st.sidebar.selectbox("Search Algorithm", ["GBFS", "A*"])

run = st.sidebar.button("🚀 Run Search")

if run:
    h = compute_heuristics(goal)

    if algorithm == "GBFS":
        path, cost, expansion = gbfs(G, start, goal, h)
    else:
        path, cost, expansion = astar(G, start, goal, h)

    if path is None:
        st.error("No path found.")
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        pos = nx.get_node_attributes(G, "pos")

        nx.draw_networkx_nodes(G, pos, node_size=1400,
                               node_color="lightblue",
                               edgecolors="black", ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=18,
                               edge_color="lightgray", width=1.8, ax=ax)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                               arrowstyle="->", arrowsize=22,
                               edge_color="red", width=3.0, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=1500,
                               node_color="orange", edgecolors="black", ax=ax)

        labels = {n: f"{n}\n{coordinates[n]}" for n in coordinates}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=9, ax=ax)

        ax.set_title(f"{algorithm}  |  {start} → {goal}",
                     fontsize=12, fontweight="bold")
        ax.grid(True)
        plt.tight_layout()

        st.pyplot(fig)

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Algorithm:** {algorithm}")
            st.write(f"**Start:** {start}")
            st.write(f"**Goal:** {goal}")
        with col2:
            st.write(f"**Solution Path:** {' → '.join(path)}")
            st.write(f"**Total Path Cost:** {cost}")
            st.write(f"**Nodes Expanded:** {len(expansion)}")

        st.write("**Expansion Order:**")
        st.write(" → ".join(expansion))