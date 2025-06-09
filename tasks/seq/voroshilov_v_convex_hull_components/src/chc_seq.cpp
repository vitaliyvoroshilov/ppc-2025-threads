#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "seq/voroshilov_v_convex_hull_components/include/chc.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::ValidationImpl() {
  int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  int height = *ptr;
  ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  int width = *ptr;
  int pixels_size = static_cast<int>(task_data->inputs_count[0]);

  return height > 0 && width > 0 && (height * width) == pixels_size;
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
  auto start = std::chrono::high_resolution_clock::now();

  Image image(height_in_, width_in_, pixels_in_);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "[SEQ Image: " << duration.count() << " ms]" << std::endl;

  start = std::chrono::high_resolution_clock::now();

  std::vector<Component> components = FindComponents(image);

  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "[SEQ FindComponents: " << duration.count() << " ms]" << std::endl;

  start = std::chrono::high_resolution_clock::now();

  hulls_out_ = QuickHullAll(components);

  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "[SEQ QuickHullAll: " << duration.count() << " ms]" << std::endl;

  return true;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::PostProcessingImpl() {
  auto start = std::chrono::high_resolution_clock::now();

  int *hulls_indxs = reinterpret_cast<int *>(task_data->outputs[0]);
  int *pixels_indxs = reinterpret_cast<int *>(task_data->outputs[1]);
  PackHulls(hulls_out_, width_in_, height_in_, hulls_indxs, pixels_indxs);
  task_data->outputs_count[0] = hulls_out_.size();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "[SEQ PostProcessing: " << duration.count() << " ms]" << std::endl;

  return true;
}
