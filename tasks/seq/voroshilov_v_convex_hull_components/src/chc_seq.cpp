#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "seq/voroshilov_v_convex_hull_components/include/chc.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::ValidationImpl() {
  int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  int height = *ptr;
  ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  int width = *ptr;
  return height > 0 && width > 0;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::PreProcessingImpl() {
  int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  height_in_ = *ptr;

  ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  width_in_ = *ptr;

  pixels_in_.resize(task_data->inputs_count[0]);
  ptr = reinterpret_cast<int *>(task_data->inputs[2]);
  std::ranges::copy(ptr, ptr + task_data->inputs_count[0], pixels_in_.begin());

  return true;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::RunImpl() {
  Image image(height_in_, width_in_, pixels_in_);
  
  std::vector<Component> components = FindComponents(image);

  std::vector<Hull> hulls = QuickHullAll(components);
  hulls_size_out = hulls.size();

  std::pair<std::vector<int>, std::vector<int>> packed_out = PackHulls(hulls, image);
  hulls_indexes_out_ = packed_out.first;
  pixels_indexes_out_ = packed_out.second;

  return true;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::PostProcessingImpl() {
  std::ranges::copy(hulls_indexes_out_, reinterpret_cast<int *>(task_data->outputs[0]));
  std::ranges::copy(pixels_indexes_out_, reinterpret_cast<int *>(task_data->outputs[1]));
  task_data->outputs_count[0] = hulls_size_out;

  return true;
}
