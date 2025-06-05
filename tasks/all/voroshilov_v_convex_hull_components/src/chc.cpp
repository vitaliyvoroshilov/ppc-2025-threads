#include "../include/chc.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace voroshilov_v_convex_hull_components_all;

Pixel::Pixel(int y_param, int x_param) : y(y_param), x(x_param), value(0) {}
Pixel::Pixel(int y_param, int x_param, int value_param) : y(y_param), x(x_param), value(value_param) {}

bool Pixel::operator==(const int value_param) const { return value == value_param; }
bool Pixel::operator==(const Pixel& other) const { return (y == other.y) && (x == other.x); }

Image::Image(int hght, int wdth, std::vector<int> pxls) {
  height = hght;
  width = wdth;
  pixels.resize(height * width);

#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      pixels[(y * width) + x] = Pixel(y, x, pxls[(y * width) + x]);
    }
  }
}

Pixel& Image::GetPixel(int y, int x) { return pixels[(y * width) + x]; }

LineSegment::LineSegment(Pixel& a_param, Pixel& b_param) : a(a_param), b(b_param) {}

UnionFind::UnionFind(int n) : roots(n), ranks(n, 1) {
  for (int i = 0; i < n; i++) {
    roots[i] = i;
  }
}

int UnionFind::FindRoot(int x) {
  if (roots[x] != x) {
    roots[x] = FindRoot(roots[x]);
  }
  return roots[x];
}

void UnionFind::Union(int x, int y) {
  int root_x = FindRoot(x);
  int root_y = FindRoot(y);
  if (root_x != root_y) {
    if (ranks[root_x] > ranks[root_y]) {
      roots[root_y] = root_x;
    } else if (ranks[root_x] < ranks[root_y]) {
      roots[root_x] = root_y;
    } else {
      roots[root_y] = root_x;
      ranks[root_x]++;
    }
  }
}

std::vector<Component> voroshilov_v_convex_hull_components_all::LabelsToComponents(std::vector<int>& labels,
                                                                                   Image& image, int num_components) {
  int height = image.height;
  int width = image.width;
  int n = height * width;

  std::unordered_map<int, std::vector<int>> groups;
  groups.reserve(num_components);

  for (int i = 0; i < n; ++i) {
    int lab = labels[i];
    if (lab > 1) {
      groups[lab].push_back(i);
    }
  }

  std::vector<Component> components;
  components.reserve(groups.size());

  for (auto& kv : groups) {
    int root_label = kv.first;
    const std::vector<int>& idxs = kv.second;
    Component comp;
    comp.reserve(idxs.size());
    for (int i : idxs) {
      int y = i / width;
      int x = i % width;
      comp.emplace_back(y, x, root_label);
    }
    components.push_back(std::move(comp));
  }
  return components;
}

void voroshilov_v_convex_hull_components_all::DepthComponentSearchInArea(std::vector<int>& labels, Image& image, int sy,
                                                                         int sx, int index, int start_y, int end_y) {
  const int step_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};  // Offsets by Y (up, stand, down)
  const int step_x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};  // Offsets by X (left, stand, right)

  std::stack<int> stack;
  int width = image.width;
  int start_index = sy * width + sx;
  labels[start_index] = index;
  stack.push(start_index);

  while (!stack.empty()) {
    int current_index = stack.top();
    stack.pop();
    int cy = current_index / width;
    int cx = current_index % width;
    for (int i = 0; i < 8; i++) {
      int ny = cy + step_y[i];
      int nx = cx + step_x[i];
      if (ny >= end_y || ny < start_y || nx >= width || nx < 0) continue;
      int next_index = ny * width + nx;
      if (image.pixels[next_index] == 1 && labels[next_index] == 0) {
        labels[next_index] = index;
        stack.push(next_index);
      }
    }
  }
}

int voroshilov_v_convex_hull_components_all::FindComponentsInArea(std::vector<int>& labels, Image& image, int start_y,
                                                                  int end_y, int index_offset) {
  int width = image.width;
  int offset = index_offset;  // unique index in this area
  int num_components = 0;

  for (int y = start_y; y < end_y; y++) {
    for (int x = 0; x < width; x++) {
      int index = y * width + x;
      if (image.pixels[index] == 1 && labels[index] == 0) {
        DepthComponentSearchInArea(labels, image, y, x, offset, start_y, end_y);
        num_components++;
        offset++;
      }
    }
  }

  return num_components;
}

std::vector<Component> voroshilov_v_convex_hull_components_all::FindComponentsOMP(Image& image) {
  int height = image.height;
  int width = image.width;
  int n = height * width;

  std::vector<int> labels(n, 0);
  int num_threads = omp_get_max_threads();

  int area_height = height / num_threads;
  int remainder = height % num_threads;
  std::vector<int> start_y(num_threads);
  std::vector<int> end_y(num_threads);
  std::vector<int> index_offset(num_threads);

  if (num_threads == 1) {
    start_y[0] = 0;
    end_y[0] = height;
    index_offset[0] = 2;
  } else {
    for (size_t i = 1; i < start_y.size(); i++) {
      start_y[i] = start_y[i - 1] + area_height;
      if (remainder > 0) {
        start_y[i]++;
        remainder--;
      }
    }

    for (size_t i = 0; i < end_y.size() - 1; i++) {
      end_y[i] = start_y[i + 1];
    }
    end_y[end_y.size() - 1] = height;

    for (int i = 0; i < num_threads; i++) {
      index_offset[i] = (i * 100000) + 2;
    }
  }

  int num_components = 0;

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();

    num_components +=
        FindComponentsInArea(labels, image, start_y[thread_id], end_y[thread_id], index_offset[thread_id]);
  }

  int max_raw_label = 0;
  for (int v : labels) {
    if (v > max_raw_label) max_raw_label = v;
  }
  UnionFind uf(max_raw_label + 1);

  for (int i = 0; i < num_threads; i++) {
    int y = end_y[i] - 1;
    if (y < 0 || y >= height - 1) continue;
    int base = y * width;
    int base_down = (y + 1) * width;
    for (int x = 0; x < width; ++x) {
      int id1 = labels[base + x];
      if (id1 <= 1) continue;
      int id2 = labels[base_down + x];
      if (id2 > 1) uf.Union(id1, id2);
      if (x > 0) {
        int id3 = labels[base_down + (x - 1)];
        if (id3 > 1) uf.Union(id1, id3);
      }
      if (x + 1 < width) {
        int id4 = labels[base_down + (x + 1)];
        if (id4 > 1) uf.Union(id1, id4);
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    if (labels[i] > 1) {
      labels[i] = uf.FindRoot(labels[i]);
    }
  }

  std::vector<Component> final_components = LabelsToComponents(labels, image, num_components);
  return final_components;
}

int voroshilov_v_convex_hull_components_all::CheckRotation(Pixel& first, Pixel& second, Pixel& third) {
  return ((second.x - first.x) * (third.y - second.y)) - ((second.y - first.y) * (third.x - second.x));
}

Pixel voroshilov_v_convex_hull_components_all::FindFarthestPixel(std::vector<Pixel>& pixels,
                                                                 LineSegment& line_segment) {
  Pixel farthest_pixel(-1, -1, -1);
  double max_dist = 0.0;

  for (Pixel& c : pixels) {
    Pixel a = line_segment.a;
    Pixel b = line_segment.b;
    if (CheckRotation(a, b, c) < 0) {  // left rotation
      double distance = std::abs(((b.x - a.x) * (a.y - c.y)) - ((a.x - c.x) * (b.y - a.y)));
      if (distance > max_dist) {
        max_dist = distance;
        farthest_pixel = c;
      }
    }
  }

  return farthest_pixel;
}

std::vector<Pixel> voroshilov_v_convex_hull_components_all::QuickHull(Component& component) {
  if (component.size() < 3) {
    return component;
  }

  Pixel left = component[0];
  Pixel right = component[0];

  for (Pixel& pixel : component) {
    if ((pixel.x < left.x) || (pixel.x == left.x && pixel.y < left.y)) {
      left = pixel;
    }
    if ((pixel.x > right.x) || (pixel.x == right.x && pixel.y > right.y)) {
      right = pixel;
    }
  }

  std::vector<Pixel> hull;
  std::stack<LineSegment> stack;

  LineSegment line_segment1(left, right);
  LineSegment line_segment2(right, left);
  stack.push(line_segment1);
  stack.push(line_segment2);

  while (!stack.empty()) {
    LineSegment line_segment = stack.top();
    Pixel a = line_segment.a;
    Pixel b = line_segment.b;
    stack.pop();

    Pixel c = FindFarthestPixel(component, line_segment);
    if (c == -1) {
      hull.push_back(a);
    } else {
      LineSegment new_line1(a, c);
      stack.push(new_line1);
      LineSegment new_line2(c, b);
      stack.push(new_line2);
    }
  }

  std::ranges::reverse(hull);

  std::vector<Pixel> res_hull;
  for (size_t i = 0; i < hull.size(); i++) {
    if (i == 0 || i == hull.size() - 1 || CheckRotation(hull[i - 1], hull[i], hull[i + 1]) != 0) {
      res_hull.push_back(hull[i]);
    }
  }

  return res_hull;
}

void voroshilov_v_convex_hull_components_all::ComputePartition(int vec_size, int world_size, std::vector<int>& parts,
                                                               std::vector<int>& offsets) {
  int base = vec_size / world_size;
  int remainder = vec_size % world_size;
  parts.resize(world_size);
  offsets.resize(world_size);
  for (int i = 0; i < world_size; i++) {
    parts[i] = base;
    if (remainder > 0) {
      parts[i]++;
      remainder--;
    }
    if (i == 0) {
      offsets[i] = 0;
    } else {
      offsets[i] = offsets[i - 1] + parts[i - 1];
    }
  }
}

std::vector<std::vector<int>> voroshilov_v_convex_hull_components_all::PackIdxs(std::vector<Component>& components,
                                                                                int image_width,
                                                                                std::vector<int>& parts,
                                                                                std::vector<int>& offsets,
                                                                                std::vector<int>& comp_sizes) {
  int world_size = static_cast<int>(parts.size());
  std::vector<std::vector<int>> split_idxs(world_size);
  for (int i = 0; i < world_size; i++) {
    int part = parts[i];
    int offset = offsets[i];
    int total_pixels = 0;
    for (int j = 0; j < part; j++) {
      total_pixels += comp_sizes[offset + j];
    }
    split_idxs[i].reserve(total_pixels);
    for (int j = 0; j < part; j++) {
      Component comp = components[offset + j];
      for (Pixel& p : comp) {
        split_idxs[i].push_back((p.y * image_width) + p.x);
      }
    }
  }
  return split_idxs;
}

std::vector<Hull> voroshilov_v_convex_hull_components_all::QuickHullAllOMP(std::vector<Component>& components) {
  if (components.empty()) {
    return {};
  }

  int components_size = static_cast<int>(components.size());
  std::vector<Hull> hulls(components.size());

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < components_size; i++) {
    hulls[i] = QuickHull(components[i]);
  }

  return hulls;
}

std::vector<Hull> voroshilov_v_convex_hull_components_all::QuickHullAllMPIOMP(std::vector<Component>& components,
                                                                              int image_width) {
  boost::mpi::communicator world;
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::broadcast(world, image_width, 0);

  std::vector<int> comp_sizes;
  if (world.rank() == 0) {
    comp_sizes.reserve(components.size());
    for (Component& comp : components) {
      comp_sizes.push_back(static_cast<int>(comp.size()));
    }
  }
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::broadcast(world, comp_sizes, 0);

  std::vector<int> parts;
  std::vector<int> offsets;
  if (world.rank() == 0) {
    ComputePartition(static_cast<int>(components.size()), world.size(), parts, offsets);
  }
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::broadcast(world, parts, 0);
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::broadcast(world, offsets, 0);

  std::vector<std::vector<int>> split_idxs;
  if (world.rank() == 0) {
    split_idxs = PackIdxs(components, image_width, parts, offsets, comp_sizes);
  }

  std::vector<int> local_idxs;
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::scatter(world, split_idxs, local_idxs, 0);

  std::vector<Component> local_components(parts[world.rank()]);
  int pos = 0;
  for (int i = 0; i < parts[world.rank()]; i++) {
    int comp_size = comp_sizes[offsets[world.rank()] + i];
    Component comp;
    comp.reserve(comp_size);
    for (int j = 0; j < comp_size; j++, pos++) {
      int idx = local_idxs[pos];
      int y = idx / image_width;
      int x = idx % image_width;
      comp.emplace_back(y, x, 1);
    }
    local_components[i] = std::move(comp);
  }

  int local_components_size = static_cast<int>(local_components.size());
  std::vector<Hull> local_hulls(local_components.size());

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < local_components_size; i++) {
    local_hulls[i] = QuickHull(local_components[i]);
  }

  std::vector<std::vector<Hull>> gathered_hulls;
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::gather(world, local_hulls, gathered_hulls, 0);
  if (world.rank() == 0) {
    std::vector<Hull> hulls;
    for (auto& vector_hulls : gathered_hulls) {
      hulls.insert(hulls.end(), std::make_move_iterator(vector_hulls.begin()),
                   std::make_move_iterator(vector_hulls.end()));
    }

    return hulls;
  }

  return {};
}

void voroshilov_v_convex_hull_components_all::PackHulls(std::vector<Hull>& hulls, int width, int height,
                                                        int* hulls_indxs, int* pixels_indxs) {
  std::fill(hulls_indxs, hulls_indxs + (height * width), 0);
  std::fill(pixels_indxs, pixels_indxs + (height * width), 0);

  int hull_index = 1;
  for (Hull& hull : hulls) {
    int pixel_index = 1;
    for (Pixel& p : hull) {
      int pos = (p.y * width) + p.x;
      hulls_indxs[pos] = hull_index;
      pixels_indxs[pos] = pixel_index;
      pixel_index++;
    }
    hull_index++;
  }
}

std::vector<Hull> voroshilov_v_convex_hull_components_all::UnpackHulls(std::vector<int>& hulls_indexes,
                                                                       std::vector<int>& pixels_indexes, int height,
                                                                       int width, size_t hulls_size) {
  std::vector<Hull> hulls(hulls_size);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int hull_index = hulls_indexes[(y * width) + x];
      if (hull_index > 0) {
        int pixel_index = pixels_indexes[(y * width) + x];
        Pixel pixel(y, x, pixel_index);
        hulls[hull_index - 1].push_back(pixel);
      }
    }
  }

  for (Hull& hull : hulls) {
    for (size_t p1 = 0; p1 < hull.size() - 1; p1++) {
      for (size_t p2 = p1 + 1; p2 < hull.size(); p2++) {
        if (hull[p1].value > hull[p2].value) {
          Pixel tmp = hull[p1];
          hull[p1] = hull[p2];
          hull[p2] = tmp;
        }
      }
    }
  }

  if (hulls.empty()) {
    return {};
  }

  return hulls;
}
