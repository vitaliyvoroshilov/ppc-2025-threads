#include "../include/chc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stack>
#include <vector>

using namespace voroshilov_v_convex_hull_components_seq;

Pixel::Pixel(int y_param, int x_param) {
  y = y_param;
  x = x_param;
  value = 0;
}
Pixel::Pixel(int y_param, int x_param, int value_param) {
  y = y_param;
  x = x_param;
  value = value_param;
}
Pixel::Pixel(const Pixel& other) {
  y = other.y;
  x = other.x;
  value = other.value;
}
Pixel& Pixel::operator=(const int value_param) {
  value = value_param;

  return *this;
}
bool Pixel::operator==(const int value_param) const { return value == value_param; }
bool Pixel::operator==(const Pixel& other) const { return (y == other.y) && (x == other.x); }
bool Pixel::operator!=(const Pixel& other) const { return !(this == &other); }

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
Image::Image(const Image& other) {
  height = other.height;
  width = other.width;
  pixels = other.pixels;
}
Pixel& Image::GetPixel(int y, int x) { return pixels[(y * width) + x]; }

void Component::AddPixel(const Pixel& pixel) { pixels.push_back(pixel); }

LineSegment::LineSegment(Pixel& a_param, Pixel& b_param) {
  a = a_param;
  b = b_param;
}

Component voroshilov_v_convex_hull_components_seq::DepthComponentSearch(Pixel& start_pixel, Image* tmp_image,
                                                                        int index) {
  const int step_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};  // Offsets by Y (up, stand, down)
  const int step_x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};  // Offsets by X (left, stand, right)
  std::stack<Pixel> stack;
  Component component;
  stack.push(start_pixel);
  tmp_image->GetPixel(start_pixel.y, start_pixel.x) = index;              // Mark start pixel as visited
  component.AddPixel(tmp_image->GetPixel(start_pixel.y, start_pixel.x));  // Add start pixel to component

  while (!stack.empty()) {
    Pixel current_pixel = stack.top();
    stack.pop();
    for (int i = 0; i < 8; i++) {
      int nextY = current_pixel.y + step_y[i];
      int nextX = current_pixel.x + step_x[i];
      if (nextY >= 0 && nextY < tmp_image->height && nextX >= 0 && nextX < tmp_image->width &&
          tmp_image->GetPixel(nextY, nextX) == 1) {
        stack.push(tmp_image->GetPixel(nextY, nextX));
        tmp_image->GetPixel(nextY, nextX) = index;              // Mark neighbour pixel as visited
        component.AddPixel(tmp_image->GetPixel(nextY, nextX));  // Add neighbour pixel to component
      }
    }
  }

  return component;
}

std::vector<Component> voroshilov_v_convex_hull_components_seq::FindComponents(Image& image) {
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
  return components;
}

int voroshilov_v_convex_hull_components_seq::CheckRotation(Pixel& first, Pixel& second, Pixel& third) {
  int rotation = ((second.x - first.x) * (third.y - second.y)) - ((second.y - first.y) * (third.x - second.x));

  int res = 0;
  if (rotation < 0) {
    res = 1;  // left rotation
  } else {
    if (rotation > 0) {
      res = -1;  // right rotation
    }
  }

  return res;
}

Pixel voroshilov_v_convex_hull_components_seq::FindLeftPixel(Component component) {
  Pixel left = component.pixels[0];

  for (Pixel& pixel : component.pixels) {
    if (pixel.x < left.x) {
      left = pixel;
    }
  }
  return left;
}

Pixel voroshilov_v_convex_hull_components_seq::FindRightPixel(Component component) {
  Pixel right = component.pixels[0];

  for (Pixel& pixel : component.pixels) {
    if (pixel.x > right.x) {
      right = pixel;
    }
  }
  return right;
}

Pixel voroshilov_v_convex_hull_components_seq::FindFarthestPixel(std::vector<Pixel>& pixels,
                                                                 LineSegment& line_segment) {
  Pixel farthest_pixel(-1, -1, -1);
  double max_dist = 0.0;

  for (Pixel& c : pixels) {
    Pixel a = line_segment.a;
    Pixel b = line_segment.b;
    if (CheckRotation(a, b, c) == 1) {  // left rotation
      double distance = std::abs(((b.x - a.x) * (a.y - c.y)) - ((a.x - c.x) * (b.y - a.y)));
      if (distance > max_dist) {
        max_dist = distance;
        farthest_pixel = c;
      }
    }
  }

  return farthest_pixel;
}

std::vector<Pixel> voroshilov_v_convex_hull_components_seq::QuickHull(Component component) {
  if (component.pixels.size() < 3) {
    return component.pixels;
  }

  Pixel left = FindLeftPixel(component);
  Pixel right = FindRightPixel(component);

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

  std::reverse(hull.begin(), hull.end());

  return hull;
}

std::vector<Hull> voroshilov_v_convex_hull_components_seq::QuickHullAll(std::vector<Component>& components) {
  std::vector<Hull> hulls;
  for (Component& component : components) {
    Hull hull;
    hull.pixels = QuickHull(component);
    hulls.push_back(hull);
  }
  return hulls;
}

std::vector<int> voroshilov_v_convex_hull_components_seq::PackHulls(std::vector<Hull>& hulls) {
  std::vector<int> packed;
  for (Hull& hull : hulls) {
    packed.push_back(hull.pixels.size());
    for (Pixel& pixel : hull.pixels) {
      packed.push_back(pixel.y);
      packed.push_back(pixel.x);
    }
  }

  return packed;
}

std::vector<Hull> voroshilov_v_convex_hull_components_seq::UnpackHulls(std::vector<int>& packed, int length) {
  std::vector<Hull> hulls;
  int i = 0;
  while (i < length) {
    int remained_in_hull = packed[i];
    i++;
    Hull hull;
    while (remained_in_hull > 0) {
      int y = packed[i];
      i++;
      int x = packed[i];
      i++;
      Pixel pixel(y, x, 0);
      hull.pixels.push_back(pixel);
      remained_in_hull--;
    }
    hulls.push_back(hull);
  }
  return hulls;
}
