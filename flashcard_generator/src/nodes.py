import uuid

def add_nodes_edges(graph, data, parent_id=None):
    """
    Recursively adds nodes and edges to a Graphviz graph for a pure topic hierarchy.

    Args:
        graph: The Graphviz graph object.
        data (dict): The dictionary representing the mind map structure.
        parent_id (str, optional): The unique ID of the parent node. Defaults to None.
    """
    for key, value in data.items():
        node_id = str(uuid.uuid4())
        graph.node(node_id, key, shape='box', style='rounded,filled', fillcolor='#cce5ff', fontcolor='#003366')

        if parent_id:
            graph.edge(parent_id, node_id)

        if isinstance(value, dict) and value:
            add_nodes_edges(graph, value, parent_id=node_id)