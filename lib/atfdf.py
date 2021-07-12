# Ardwino TensorFlow Decision Forests
# Mathieu Guillame-Bert, 2021
#
# Export TensorFlow Decision Forest models to Ardwino

import tensorflow_decision_forests as tfdf
import struct

# Format of the model.
FORMAT_VERSION = 1


def convert_model(model, output_path):
    """Converts a TensorFlow Decision Forests model into an Arduino compatible serving model.

    The input model format is described at: https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/py_tree
    The output model format is described in the "Binary model format" section of the `README.md`.
    """

    # Access the model structure.
    inspector = model.make_inspector()

    # The initial prediction is the value of the model without any tree. It is also called "bias". The initial prediction value is added to the leaf outputs
    # of the first tree.
    initial_prediction = inspector.specialized_header().initial_predictions[0]

    # Each tree in the binary output format.
    forest_data = []

    # Total memory usage of the forest data. Sum of len of "forest_data".
    forest_data_size_in_bytes = 0

    # Maximum binary format size among the tree. Used to determine the required inference buffer size.
    max_tree_data_size_in_bytes = 0

    # Total number of nodes in the model.
    total_num_nodes = 0

    # Number of nodes for each tree.
    num_node_per_trees = []

    for tree_idx in range(inspector.num_trees()):

        # Extract a tree.
        # Patch bug in TF-DF v0.1.7.
        requested_tree_idx = tree_idx
        if tfdf.__version__ == "0.1.7":
            requested_tree_idx = tree_idx + 1
        tree = inspector.extract_tree(tree_idx=requested_tree_idx)

        # Convert it to the binary format.
        tree_data = bytearray()
        num_nodes = _export_node(tree.root, tree_data,
                                 initial_prediction if tree_idx == 0 else 0)
        forest_data.append(tree_data)
        num_node_per_trees.append(num_nodes)

        # Update the various metrics.
        total_num_nodes += num_nodes
        tree_data_size_in_bytes = len(tree_data)
        forest_data_size_in_bytes += tree_data_size_in_bytes
        if tree_data_size_in_bytes > max_tree_data_size_in_bytes:
            max_tree_data_size_in_bytes = tree_data_size_in_bytes

    # Model header.
    header = struct.pack("<HHH", FORMAT_VERSION,
                         inspector.num_trees(), total_num_nodes)

    print(f"The model takes {forest_data_size_in_bytes} bytes of Flash memory and {max_tree_data_size_in_bytes + len(header)} bytes of RAM memory (during inference).")

    # Export the model data to a .h file.
    with open(output_path, "w") as output_file:

        # Comments.
        output_file.write("// Exported TensorFlow Decision Forests model for Arduino.\n")
        output_file.write("// This file was automatically generated.\n\n")

        output_file.write("#include \"atfdf.h\"\n\n")

        output_file.write(f"// Binary format version: {FORMAT_VERSION}\n")
        output_file.write(
            f"// Flash memory usage: {forest_data_size_in_bytes} bytes\n")
        output_file.write(
            f"// RAM usage during inference: {max_tree_data_size_in_bytes + len(header)} bytes\n")
        output_file.write(
            f"// Input features ({len(inspector.features())}):\n")
        for feature_idx, feature in enumerate(inspector.features()):
            output_file.write(f"// \t{feature_idx}: {feature}\n")

        # Individual tree data store in the flash memory.
        tree_variable_names = []
        for tree_idx, tree_data in enumerate(forest_data):
            variable_name = f"_kModelTree{tree_idx}"
            _write_bytearray_to_cc_variable(
                output_file, variable_name, True, tree_data)
            tree_variable_names.append(variable_name)

        # Model object.
        output_file.write("Model kMyModel {/*header=*/ \"")
        _write_bytearray_to_cc_char_array(output_file, header)
        output_file.write("\"\n\t, /*buffer_size=*/ ")
        output_file.write(str(max_tree_data_size_in_bytes))
        output_file.write("\n\t, /*num_nodes=*/ (uint16_t[]){")
        output_file.write(", ".join(map(str, num_node_per_trees)))
        output_file.write("}\n\t, (const char *const []){")
        output_file.write(", ".join(tree_variable_names))
        output_file.write("}};\n")

    print(f"Model data exported to: {output_path}")


def _write_bytearray_to_cc_char_array(output_file, data: bytearray):
    for value in data:
        output_file.write("\\x" + format(value, "02x"))


def _write_bytearray_to_cc_variable(output_file, variable_name: str, prog_mem: bool, data: bytearray):
    output_file.write("const char ")
    if prog_mem:
        output_file.write("PROGMEM ")
    output_file.write(variable_name)
    output_file.write("[] = \"")
    for value_idx, value in enumerate(data):
        if ((value_idx+1) % 20) == 0:
            # Insert new line.
            output_file.write("\"\n\t\"")
        output_file.write("\\x" + format(value, "02x"))
    output_file.write("\";\n")


def _export_node(node: tfdf.py_tree.node.AbstractNode, dst: bytearray, value_offset: float) -> int:
    """Export the binary data for a node and its children.

    Args:
        node: Node to export.
        dst: Binary destination.
        value_offset: Value to add to each leaf value.

    Returns:
        The number of exported nodes.
    """

    if isinstance(node, tfdf.py_tree.node.NonLeafNode):
        # Non leaf node i.e. condition node.

        if isinstance(node.condition, tfdf.py_tree.condition.NumericalHigherThanCondition):
            save_node_offset_pos = len(dst)
            # Node data.
            dst.extend(struct.pack("<HHf",
                                   0,
                                   node.condition.feature.col_idx,
                                   node.condition.threshold))

            # Export children.
            neg_nodes = _export_node(node.neg_child, dst, value_offset)
            pos_nodes = _export_node(node.pos_child, dst, value_offset)

            # Update the "positive_child_offset".
            dst[save_node_offset_pos:(
                save_node_offset_pos+2)] = struct.pack("H", neg_nodes)

            return 1 + pos_nodes + neg_nodes

        else:
            raise ValueError(f"Non supported condition: {node.condition}")

    else:
        # Leaf node
        dst.extend(struct.pack("<HHf",
                               0,
                               0,
                               node.value.value + value_offset))
        return 1
