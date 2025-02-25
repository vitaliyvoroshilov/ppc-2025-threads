#include "../include/chc.hpp"

#include <algorithm>
#include <cmath>
#include <stack>
#include <vector>

using namespace voroshilov_v_convex_hull_components_seq;

Pixel::Pixel(int yParam, int xParam) {
  y = yParam;
  x = xParam;
  value = 0;
}
Pixel::Pixel(int yParam, int xParam, int valueParam) {
  y = yParam;
  x = xParam;
  value = valueParam;
}
Pixel::Pixel(const Pixel& other) {
  y = other.y;
  x = other.x;
  value = other.value;
}
Pixel& Pixel::operator=(const int valueParam) {
  value = valueParam;

  return *this;
}
bool Pixel::operator==(const int valueParam) const { return value == valueParam; }
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
Pixel& Image::getPixel(int y, int x) { return pixels[(y * width) + x]; }

void Component::addPixel(const Pixel& pixel) { pixels.push_back(pixel); }

LineSegment::LineSegment(Pixel aParam, Pixel bParam) {
  a = aParam;
  b = bParam;
}

Component voroshilov_v_convex_hull_components_seq::depthComponentSearch(Pixel& startPixel, Image* tmpImage, int index) {
  const int stepY[8] = {1, 1, 1, 0, 0, -1, -1, -1};  // Offsets by Y (up, stand, down)
  const int stepX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};  // Offsets by X (left, stand, right)
  std::stack<Pixel> stack;
  Component component;
  stack.push(startPixel);
  tmpImage->getPixel(startPixel.y, startPixel.x) = index;              // Mark start pixel as visited
  component.addPixel(tmpImage->getPixel(startPixel.y, startPixel.x));  // Add start pixel to component

  while (!stack.empty()) {
    Pixel currentPixel = stack.top();
    stack.pop();
    for (int i = 0; i < 8; i++) {
      int nextY = currentPixel.y + stepY[i];
      int nextX = currentPixel.x + stepX[i];
      if (nextY >= 0 && nextY < tmpImage->height && nextX >= 0 && nextX < tmpImage->width &&
          tmpImage->getPixel(nextY, nextX) == 1) {
        stack.push(tmpImage->getPixel(nextY, nextX));
        tmpImage->getPixel(nextY, nextX) = index;              // Mark neighbour pixel as visited
        component.addPixel(tmpImage->getPixel(nextY, nextX));  // Add neighbour pixel to component
      }
    }
  }

  return component;
}

std::vector<Component> voroshilov_v_convex_hull_components_seq::findComponents(Image& image) {
  Image tmpImage(image);
  std::vector<Component> components;
  int count = 0;
  for (int y = 0; y < tmpImage.height; y++) {
    for (int x = 0; x < tmpImage.width; x++) {
      if (tmpImage.getPixel(y, x) == 1) {
        Component component = depthComponentSearch(tmpImage.getPixel(y, x), &tmpImage, count + 2);
        components.push_back(component);
        count++;
      }
    }
  }
  return components;
}

int voroshilov_v_convex_hull_components_seq::checkRotation(Pixel& first, Pixel& second, Pixel& third) {
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

Pixel voroshilov_v_convex_hull_components_seq::findLeftPixel(Component component) {
  Pixel left = component.pixels[0];

  for (Pixel& pixel : component.pixels) {
    if (pixel.x < left.x) {
      left = pixel;
    }
  }
  return left;
}

Pixel voroshilov_v_convex_hull_components_seq::findRightPixel(Component component) {
  Pixel right = component.pixels[0];

  for (Pixel& pixel : component.pixels) {
    if (pixel.x > right.x) {
      right = pixel;
    }
  }
  return right;
}

Pixel voroshilov_v_convex_hull_components_seq::findFarthestPixel(std::vector<Pixel>& pixels, LineSegment lineSegment) {
  Pixel farthestPixel(-1, -1, -1);
  double maxDist = 0.0;

  for (Pixel& c : pixels) {
    Pixel a = lineSegment.a;
    Pixel b = lineSegment.b;
    if (checkRotation(a, b, c) == 1) {  // left rotation
      double distance = std::abs(((b.x - a.x) * (a.y - c.y)) - ((a.x - c.x) * (b.y - a.y)));
      if (distance > maxDist) {
        maxDist = distance;
        farthestPixel = c;
      }
    }
  }

  return farthestPixel;
}

std::vector<Pixel> voroshilov_v_convex_hull_components_seq::quickHull(Component component) {
  if (component.pixels.size() < 3) {
    return component.pixels;
  }

  Pixel left = findLeftPixel(component);
  Pixel right = findRightPixel(component);

  std::vector<Pixel> hull;
  std::stack<LineSegment> stack;

  LineSegment lineSegment1(left, right);
  LineSegment lineSegment2(right, left);
  stack.push(lineSegment1);
  stack.push(lineSegment2);

  while (!stack.empty()) {
    LineSegment lineSegment = stack.top();
    Pixel a = lineSegment.a;
    Pixel b = lineSegment.b;
    stack.pop();

    Pixel c = findFarthestPixel(component.pixels, lineSegment);
    if (c == -1) {
      hull.push_back(a);
    } else {
      LineSegment newLine1(a, c);
      stack.push(newLine1);
      LineSegment newLine2(c, b);
      stack.push(newLine2);
    }
  }

  std::reverse(hull.begin(), hull.end());

  return hull;
}

std::vector<Hull> voroshilov_v_convex_hull_components_seq::quickHullAll(std::vector<Component>& components) {
  std::vector<Hull> hulls;
  for (Component& component : components) {
    Hull hull;
    hull.pixels = quickHull(component);
    hulls.push_back(hull);
  }
  return hulls;
}

std::vector<int> voroshilov_v_convex_hull_components_seq::packHulls(std::vector<Hull>& hulls) {
  std::vector<int> packed;
  for (Hull& hull : hulls) {
    packed.push_back(hull.pixels.size());
    for (Pixel pixel : hull.pixels) {
      packed.push_back(pixel.y);
      packed.push_back(pixel.x);
    }
  }

  return packed;
}

std::vector<Hull> voroshilov_v_convex_hull_components_seq::unpackHulls(std::vector<int>& packed, int length) {
  std::vector<Hull> hulls;
  int i = 0;
  while (i < length) {
    int remainedInHull = packed[i];
    i++;
    Hull hull;
    while (remainedInHull > 0) {
      int y = packed[i];
      i++;
      int x = packed[i];
      i++;
      Pixel pixel(y, x, 0);
      hull.pixels.push_back(pixel);
      remainedInHull--;
    }
    hulls.push_back(hull);
  }
  return hulls;
}
