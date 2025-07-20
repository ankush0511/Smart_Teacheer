import graphviz
from flashcard_generator.src.nodes import add_nodes_edges

def visualize_mind_map(mind_map_data: dict, output_filename: str = "mind_map") -> str:
    """
    Generates a PNG image of the mind map using Graphviz.
    """
    dot = graphviz.Digraph('MindMap', comment='Study Mind Map')
    dot.attr(rankdir='TB', splines='ortho', concentrate='true', newrank='true', size="20,20")
    dot.attr('node', fontname='Helvetica', fontsize='12', margin='0.25')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    add_nodes_edges(dot, mind_map_data)

    output_path = dot.render(output_filename, format='png', view=False, cleanup=True)
    return output_path
