#include "../include/chc_all.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <iostream>
#include <utility>
#include <vector>

#include "../include/chc.hpp"

using namespace voroshilov_v_convex_hull_components_all;

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    auto start = std::chrono::high_resolution_clock::now();

    int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    int height = *ptr;
    ptr = reinterpret_cast<int *>(task_data->inputs[1]);
    int width = *ptr;
    int pixels_size = static_cast<int>(task_data->inputs_count[0]);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "\n Proc" << world.rank() << ", Validation: " << duration << " ms \n";

    return height > 0 && width > 0 && (height * width) == pixels_size;
  }
  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto start = std::chrono::high_resolution_clock::now();

    int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    int height = *ptr;

    ptr = reinterpret_cast<int *>(task_data->inputs[1]);
    int width = *ptr;

    std::vector<int> pixels(task_data->inputs_count[0]);
    ptr = reinterpret_cast<int *>(task_data->inputs[2]);
    std::ranges::copy(ptr, ptr + task_data->inputs_count[0], pixels.begin());

    Image image(height, width, pixels);
    imageIn_ = image;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "\n Proc" << world.rank() << ", PreProcessing: " << duration << " ms \n";
  }
  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::RunImpl() {
  std::vector<Component> components;

  auto start = std::chrono::high_resolution_clock::now();

  if (world_.rank() == 0) {
    components = FindComponentsOMP(imageIn_);
  }

  hullsOut_ = QuickHullAllMPIOMP(components, imageIn_.width);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "\n Proc" << world.rank() << ", Run: " << duration << " ms \n";

  return true;
}

bool voroshilov_v_convex_hull_components_all::ChcTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto start = std::chrono::high_resolution_clock::now();

    std::pair<std::vector<int>, std::vector<int>> packed_out = PackHulls(hullsOut_, imageIn_);
    std::vector<int> hulls_indexes = packed_out.first;
    std::vector<int> pixels_indexes = packed_out.second;

    std::ranges::copy(hulls_indexes, reinterpret_cast<int *>(task_data->outputs[0]));
    std::ranges::copy(pixels_indexes, reinterpret_cast<int *>(task_data->outputs[1]));
    task_data->outputs_count[0] = hullsOut_.size();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "\n Proc" << world.rank() << ", PostProcessing: " << duration << " ms \n";
  }
  return true;
}
