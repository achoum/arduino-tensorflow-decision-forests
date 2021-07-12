// Ardwino TensorFlow Decision Forests
// Mathieu Guillame-Bert, 2021
//
// Run a TensorFlow Decision Forest model on Ardwino.

#ifndef ATFDF_H_
#define ATFDF_H_

#include "alloca.h"
#include <avr/pgmspace.h>

// Format of the model. Make sure to use an exporter with the same version.
const int kFormatVersion = 1;

struct Model {
  const char *header;
  const int buffer_size;
  const uint16_t *num_nodes;
  const char *const *trees;
};

struct Header {
  uint16_t version;
  uint16_t num_trees;
  uint16_t num_nodes;
};

struct Node {
  uint16_t positive_child_offset;
  uint16_t attribute;
  union {
    float value;
    float threshold;
  };
};

// Runs the model on an example and returns the prediction.
//
// Args:
//   example: List of attribute values making one example. See the model export
//   for the feature order.
// "example" is a list
// Returns NaN in case of an error e.g. wrong model version.
float predict(const float *example, const Model &model) {

  const Header *header = (Header *)model.header;

  // Cherck the model version.
  if (header->version != kFormatVersion) {
    // Wrong version.
    return NAN;
  }

  // Allocate a working buffer.
  Node *nodes = alloca(model.buffer_size);
  if (!nodes) {
    // Not enought RAM.
    return NAN;
  }

  float prediction_accumulator = 0.f;

  const unsigned int num_trees = header->num_trees;
  for (unsigned int tree_idx = 0; tree_idx < num_trees; tree_idx++) {

    // Load the tree data from Flash to Ram.
    memcpy_P(nodes, (Node *)model.trees[tree_idx],
             model.num_nodes[tree_idx] * sizeof(Node));

    // Follow the inference path of the decision tree.
    unsigned int node_idx = 0;
    while (true) {
      const Node *node = &nodes[node_idx];

      if (node->positive_child_offset == 0) {
        // Leaf node.
        prediction_accumulator += node->value;
        break;
      }

      // Non leaf node.
      const bool evaluation = example[node->attribute] >= node->threshold;
      if (evaluation) {
        node_idx += node->positive_child_offset + 1;
      } else {
        node_idx++;
      }
    }
  }

  return prediction_accumulator;
}

#endif // ATFDF_H_
