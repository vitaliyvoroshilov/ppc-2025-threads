#include "../include/chc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stack>
#include <unordered_map>
#include <vector>

using namespace voroshilov_v_convex_hull_components_seq;

Pixel::Pixel(int y_param, int x_param) : y(y_param), x(x_param), value(0) {}
Pixel::Pixel(int y_param, int x_param, int value_param) : y(y_param), x(x_param), value(value_param) {}

bool Pixel::operator==(const int value_param) const { return value == value_param; }
bool Pixel::operator==(const Pixel& other) const { return (y == other.y) && (x == other.x); }

Image::Image(int hght, int wdth, std::vector<int> pxls) {
  height = hght;
  width = wdth;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      Pixel pixel(y, x, pxls[(y * width) + x]);
      pixels.push_back(pixel);
    }
  }
}

Pixel& Image::GetPixel(int y, int x) { return pixels[(y * width) + x]; }

void Component::AddPixel(const Pixel& pixel) { pixels.push_back(pixel); }

LineSegment::LineSegment(Pixel& a_param, Pixel& b_param) : a(a_param), b(b_param) {}

bool Hull::operator==(const Hull& other) const { return pixels == other.pixels; }

std::vector<Component> voroshilov_v_convex_hull_components_seq::LabelsToComponents(std::vector<int>& labels,
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
    comp.pixels.reserve(idxs.size());
    for (int i : idxs) {
      int y = i / width;
      int x = i % width;
      comp.pixels.emplace_back(y, x, root_label);
    }
    components.push_back(std::move(comp));
  }
  return components;
}

void voroshilov_v_convex_hull_components_seq::DepthComponentSearch(std::vector<int>& labels, Image& image, int sy,
                                                                   int sx, int index) {
  const int step_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};  // Offsets by Y (up, stand, down)
  const int step_x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};  // Offsets by X (left, stand, right)

  std::stack<int> stack;
  int height = image.height;
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
      if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
      int next_index = ny * width + nx;
      if (image.pixels[next_index] == 1 && labels[next_index] == 0) {
        labels[next_index] = index;
        stack.push(next_index);
      }
    }
  }
}

std::vector<Component> voroshilov_v_convex_hull_components_seq::FindComponents(Image& image) {
  int height = image.height;
  int width = image.width;
  int n = height * width;
  int num_components = 0;

  std::vector<int> labels(n, 0);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int index = y * width + x;
      if (image.pixels[index] == 1 && labels[index] == 0) {
        DepthComponentSearch(labels, image, y, x, num_components + 2);
        num_components++;
      }
    }
  }

  std::vector<Component> final_components = LabelsToComponents(labels, image, num_components);

  return final_components;
}

int voroshilov_v_convex_hull_components_seq::CheckRotation(Pixel& first, Pixel& second, Pixel& third) {
  return ((second.x - first.x) * (third.y - second.y)) - ((second.y - first.y) * (third.x - second.x));
}

Pixel voroshilov_v_convex_hull_components_seq::FindFarthestPixel(std::vector<Pixel>& pixels,
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

std::vector<Pixel> voroshilov_v_convex_hull_components_seq::QuickHull(Component& component) {
  if (component.pixels.size() < 3) {
    return component.pixels;
  }

  Pixel left = component.pixels[0];
  Pixel right = component.pixels[0];

  for (Pixel& pixel : component.pixels) {
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
    if (i == 0 || i == hull.size() - 1 || CheckRotation(hull[i - 1], hull[i], hull[i + 1]) != 0) {
      res_hull.push_back(hull[i]);
    }
  }

  return res_hull;
}

std::vector<Hull> voroshilov_v_convex_hull_components_seq::QuickHullAll(std::vector<Component>& components) {
  if (components.empty()) {
    return {};
  }
  std::vector<Hull> hulls;
  for (Component& component : components) {
    Hull hull;
    hull.pixels = QuickHull(component);
    hulls.push_back(hull);
  }
  return hulls;
}

void voroshilov_v_convex_hull_components_seq::PackHulls(std::vector<Hull>& hulls, int width, int height,
                                                        int* hulls_indxs, int* pixels_indxs) {
  std::fill(hulls_indxs, hulls_indxs + (height * width), 0);
  std::fill(pixels_indxs, pixels_indxs + (height * width), 0);

  int hull_index = 1;
  for (Hull& hull : hulls) {
    int pixel_index = 1;
    for (Pixel& p : hull.pixels) {
      int pos = (p.y * width) + p.x;
      hulls_indxs[pos] = hull_index;
      pixels_indxs[pos] = pixel_index;
      pixel_index++;
    }
    hull_index++;
  }
}

std::vector<Hull> voroshilov_v_convex_hull_components_seq::UnpackHulls(std::vector<int>& hulls_indexes,
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

  return hulls;
}
