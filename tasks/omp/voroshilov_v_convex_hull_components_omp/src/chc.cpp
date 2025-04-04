#include "../include/chc.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace voroshilov_v_convex_hull_components_omp;

Pixel::Pixel(int y_param, int x_param) : y(y_param), x(x_param), value(0) {}
NOINLINE Pixel::Pixel(int y_param, int x_param, int value_param) : y(y_param), x(x_param), value(value_param) {}

NOINLINE bool Pixel::operator==(const int value_param) const { return value == value_param; }
NOINLINE bool Pixel::operator==(const Pixel& other) const { return (y == other.y) && (x == other.x); }

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

NOINLINE Pixel& Image::GetPixel(int y, int x) { return pixels[(y * width) + x]; }

Component::Component() { pixels.resize(0); }

//Component::Component(std::vector<Pixel>& pxls) { pixels = pxls; }

LineSegment::LineSegment(Pixel& a_param, Pixel& b_param) : a(a_param), b(b_param) {}

Hull::Hull() {}

Hull::Hull(std::vector<Pixel>& pxls) { pixels = pxls; }

bool Hull::operator==(const Hull& other) const { return pixels == other.pixels; }

Component voroshilov_v_convex_hull_components_omp::DepthComponentSearch(Pixel& start_pixel, Image* tmp_image,
                                                                        int index) {
  const int step_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};  // Offsets by Y (up, stand, down)
  const int step_x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};  // Offsets by X (left, stand, right)
  std::stack<Pixel> stack;
  std::vector<Pixel> component_pixels;
  stack.push(start_pixel);
  tmp_image->GetPixel(start_pixel.y, start_pixel.x).value = index;                // Mark start pixel as visited
  component_pixels.push_back(tmp_image->GetPixel(start_pixel.y, start_pixel.x));  // Add start pixel to component

  while (!stack.empty()) {
    Pixel current_pixel = stack.top();
    stack.pop();
    for (int i = 0; i < 8; i++) {
      int next_y = current_pixel.y + step_y[i];
      int next_x = current_pixel.x + step_x[i];
      if (next_y >= 0 && next_y < tmp_image->height && next_x >= 0 && next_x < tmp_image->width &&
          tmp_image->GetPixel(next_y, next_x) == 1) {
        stack.push(tmp_image->GetPixel(next_y, next_x));
        tmp_image->GetPixel(next_y, next_x).value = index;                // Mark neighbour pixel as visited
        component_pixels.push_back(tmp_image->GetPixel(next_y, next_x));  // Add neighbour pixel to component
      }
    }
  }

  Component component(component_pixels);

  return component;
}

std::vector<Component> voroshilov_v_convex_hull_components_omp::FindComponentsSeq(Image& image) {
  Image tmp_image(image);
  std::vector<Component> components;
  int count = 0;
  for (int y = 0; y < tmp_image.height; y++) {
    for (int x = 0; x < tmp_image.width; x++) {
      if (tmp_image.GetPixel(y, x) == 1) {
        Component component = DepthComponentSearch(tmp_image.GetPixel(y, x), &tmp_image, count + 2);
        components.push_back(component);
        count++;
      }
    }
  }
  if (components.empty()) {
    return {};
  }
  return components;
}

int UnionFind::FindRoot(int x) {
  if (roots.find(x) == roots.end()) {
    roots[x] = x;
    ranks[x] = 1;
  }
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

void voroshilov_v_convex_hull_components_omp::CheckBoundaryPixels(UnionFind* union_find, Image& image, int y, int x) {
  Pixel p1 = image.GetPixel(y, x);

  Pixel p2 = image.GetPixel(y + 1, x);
  if (p1.value > 1 && p2.value > 1) {
    union_find->Union(p1.value, p2.value);
  }

  if (x > 0) {
    Pixel p3 = image.GetPixel(y + 1, x - 1);
    if (p1.value > 1 && p3.value > 1) {
      union_find->Union(p1.value, p3.value);
    }
  }
  if (x < image.width - 1) {
    Pixel p4 = image.GetPixel(y + 1, x + 1);
    if (p1.value > 1 && p4.value > 1) {
      union_find->Union(p1.value, p4.value);
    }
  }
}

void voroshilov_v_convex_hull_components_omp::MergeComponentsAcrossAreas(std::vector<Component>& components,
                                                                         Image& image, int area_height,
                                                                         std::vector<int>& end_y) {
  UnionFind union_find;

  int width = image.width;
  int height = image.height;

  for (int endy : end_y) {
    int y = endy - 1;
    if (y != height - 1) {
      for (int x = 0; x < width; x++) {
        CheckBoundaryPixels(&union_find, image, y, x);
      }
    }
  }

  std::unordered_map<int, Component> merged_components;
  for (Component& component : components) {
    int new_id = union_find.FindRoot(component.pixels[0].value);
    if (merged_components.find(new_id) == merged_components.end()) {
      merged_components[new_id] = Component();
    }
    merged_components[new_id].pixels.insert(merged_components[new_id].pixels.end(), component.pixels.begin(),
                                            component.pixels.end());
  }

  components.clear();
  for (auto& entry : merged_components) {
    components.push_back(entry.second);
  }
}

Component voroshilov_v_convex_hull_components_omp::DepthComponentSearchInArea(Pixel start_pixel, Image* tmp_image,
                                                                              int index, int start_y, int end_y) {
  const int step_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};  // Offsets by Y (up, stand, down)
  const int step_x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};  // Offsets by X (left, stand, right)
  std::stack<Pixel> stack;
  std::vector<Pixel> component_pixels;
  stack.push(start_pixel);
  tmp_image->GetPixel(start_pixel.y, start_pixel.x).value = index;                // Mark start pixel as visited
  component_pixels.push_back(tmp_image->GetPixel(start_pixel.y, start_pixel.x));  // Add start pixel to component

  while (!stack.empty()) {
    Pixel current_pixel = stack.top();
    stack.pop();
    for (int i = 0; i < 8; i++) {
      int next_y = current_pixel.y + step_y[i];
      int next_x = current_pixel.x + step_x[i];
      if (next_y >= start_y && next_y < end_y && next_x >= 0 && next_x < tmp_image->width &&
          tmp_image->GetPixel(next_y, next_x) == 1) {
        stack.push(tmp_image->GetPixel(next_y, next_x));
        tmp_image->GetPixel(next_y, next_x).value = index;                // Mark neighbour pixel as visited
        component_pixels.push_back(tmp_image->GetPixel(next_y, next_x));  // Add neighbour pixel to component
      }
    }
  }

  Component component(component_pixels);

  return component;
}

std::vector<Component> voroshilov_v_convex_hull_components_omp::FindComponentsInArea(Image& tmp_image, int start_y,
                                                                                     int end_y, int index_offset) {
  std::vector<Component> components;
  int index = index_offset;  // unique index in this area

  for (int y = start_y; y < end_y; y++) {
    for (int x = 0; x < tmp_image.width; x++) {
      if (tmp_image.GetPixel(y, x) == 1) {
        Component component = DepthComponentSearchInArea(tmp_image.GetPixel(y, x), &tmp_image, index, start_y, end_y);
        components.push_back(component);
        index++;
      }
    }
  }

  if (components.empty()) {
    return {};
  }

  return components;
}

std::vector<Component> voroshilov_v_convex_hull_components_omp::FindComponentsOMP(Image& image) {
  Image tmp_image(image);

  int num_threads = omp_get_max_threads();

  std::cout << "\n FindComponentsOMP(): omp_get_max_threads = " << num_threads << "\n\n";

  std::vector<std::vector<Component>> thread_components(num_threads);

  int height = tmp_image.height;

  if (num_threads > height) {
    return FindComponentsSeq(image);
  }

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

#pragma omp parallel
  {
#pragma omp single
    {
      std::cout << "\n parallel section in FindComponentsOMP(): omp_get_num_threads = " << omp_get_num_threads()
                << "\n\n";
    }
    int thread_id = omp_get_thread_num();

    thread_components[thread_id] =
        FindComponentsInArea(tmp_image, start_y[thread_id], end_y[thread_id], index_offset[thread_id]);
  }

  std::vector<Component> components;
  for (std::vector<Component>& vec : thread_components) {
    components.insert(components.end(), vec.begin(), vec.end());
  }

  MergeComponentsAcrossAreas(components, tmp_image, area_height, end_y);

  int size = static_cast<int>(components.size());
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < size; i++) {
    std::ranges::sort(components[i].pixels,
                      [](const Pixel& p1, const Pixel& p2) { return (p1.y < p2.y || (p1.y == p2.y && p1.x < p2.x)); });
  }

  return components;
}

NOINLINE double voroshilov_v_convex_hull_components_omp::CheckRotation(Pixel& first, Pixel& second, Pixel& third) {
  return ((second.x - first.x) * (third.y - second.y)) - ((second.y - first.y) * (third.x - second.x));
}

Pixel voroshilov_v_convex_hull_components_omp::FindFarthestPixel(std::vector<Pixel>& pixels,
                                                                 LineSegment& line_segment) {
  Pixel farthest_pixel(-1, -1, -1);
  double max_dist = 0.0;

  for (Pixel& c : pixels) {
    Pixel a = line_segment.a;
    Pixel b = line_segment.b;
    if (CheckRotation(a, b, c) < 0.0) {  // left rotation
      double distance = std::abs(((b.x - a.x) * (a.y - c.y)) - ((a.x - c.x) * (b.y - a.y)));
      if (distance > max_dist) {
        max_dist = distance;
        farthest_pixel = c;
      }
    }
  }

  return farthest_pixel;
}

std::vector<Pixel> voroshilov_v_convex_hull_components_omp::QuickHull(Component& component) {
  if (component.pixels.size() < 3) {
    return component.pixels;
  }

  Pixel left = component.pixels[0];
  Pixel right = component.pixels[0];

  for (Pixel& pixel : component.pixels) {
    if (pixel.x < left.x) {
      left = pixel;
    }
    if (pixel.x > right.x) {
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

    Pixel c = FindFarthestPixel(component.pixels, line_segment);
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
    if (i == 0 || i == hull.size() - 1 || CheckRotation(hull[i - 1], hull[i], hull[i + 1]) != 0.0) {
      res_hull.push_back(hull[i]);
    }
  }

  return res_hull;
}

std::vector<Hull> voroshilov_v_convex_hull_components_omp::QuickHullAllOMP(std::vector<Component>& components) {
  if (components.empty()) {
    return {};
  }

  int components_size = static_cast<int>(components.size());
  std::vector<Hull> hulls(components.size());

  std::cout << "\n QuickHullAllOMP(): omp_get_max_threads = " << omp_get_max_threads() << "\n\n";

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < components_size; i++) {
    hulls[i].pixels = QuickHull(components[i]);
  }

  return hulls;
}

std::pair<std::vector<int>, std::vector<int>> voroshilov_v_convex_hull_components_omp::PackHulls(
    std::vector<Hull>& hulls, Image& image) {
  int height = image.height;
  int width = image.width;

  std::vector<int> hulls_indexes(height * width, 0);
  std::vector<int> pixels_indexes(height * width, 0);

  int hulls_size = static_cast<int>(hulls.size());
  std::atomic<int> uniq_hull_index(1);

#pragma omp parallel for
  for (int i = 0; i < hulls_size; i++) {
    int pixel_index = 1;
    int pixels_size = static_cast<int>(hulls[i].pixels.size());
    int hull_index = uniq_hull_index.fetch_add(1);

    for (int j = 0; j < pixels_size; j++) {
      hulls_indexes[(hulls[i].pixels[j].y * width) + hulls[i].pixels[j].x] = hull_index;
      pixels_indexes[(hulls[i].pixels[j].y * width) + hulls[i].pixels[j].x] = pixel_index;
      pixel_index++;
    }
  }

  std::pair<std::vector<int>, std::vector<int>> packed_vectors(hulls_indexes, pixels_indexes);
  return packed_vectors;
}

std::vector<Hull> voroshilov_v_convex_hull_components_omp::UnpackHulls(std::vector<int>& hulls_indexes,
                                                                       std::vector<int>& pixels_indexes, int height,
                                                                       int width, size_t hulls_size) {
  std::vector<Hull> hulls(hulls_size);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int hull_index = hulls_indexes[(y * width) + x];
      if (hull_index > 0) {
        int pixel_index = pixels_indexes[(y * width) + x];
        Pixel pixel(y, x, pixel_index);
        hulls[hull_index - 1].pixels.push_back(pixel);
      }
    }
  }

  for (Hull& hull : hulls) {
    for (size_t p1 = 0; p1 < hull.pixels.size() - 1; p1++) {
      for (size_t p2 = p1 + 1; p2 < hull.pixels.size(); p2++) {
        if (hull.pixels[p1].value > hull.pixels[p2].value) {
          Pixel tmp = hull.pixels[p1];
          hull.pixels[p1] = hull.pixels[p2];
          hull.pixels[p2] = tmp;
        }
      }
    }
  }

  if (hulls.empty()) {
    return {};
  }

  return hulls;
}
