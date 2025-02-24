#pragma once

#include <vector>

namespace voroshilov_v_convex_hull_components_seq {

struct Pixel {
  int y;      // vertical coordinate, 0 is top
  int x;      // horizontal coordinate, 0 is left
  int value;  // the pixel itself

  Pixel() = default;
  Pixel(int y_, int x_);
  Pixel(int y_, int x_, int value_);
  Pixel(const Pixel& other);
  Pixel& operator=(const Pixel& other);
  Pixel& operator=(const int value_);
  bool operator==(const int value_) const;
  bool operator==(const Pixel& other) const;
  bool operator!=(const Pixel& other) const;
};

struct Image {
  int height;
  int width;
  std::vector<Pixel> pixels;

  Image() = default;
  Image(int height_, int width_, std::vector<int> pixels_);
  Image(const Image& other);
  Image& operator=(const Image& other);
  Pixel& getPixel(int y, int x);
  std::vector<int> getValues() const;
};

struct Component {
  std::vector<Pixel> pixels;

  void addPixel(Pixel pixel);
};

struct LineSegment {
  Pixel a;
  Pixel b;

  LineSegment(Pixel a_, Pixel b_);
};

struct Hull {
  std::vector<Pixel> pixels;

  Hull() = default;
};

Component depthComponentSearch(Pixel startPixel, Image* tmpImage, int index);

std::vector<Component> findComponents(Image image);

int checkRotation(Pixel first, Pixel second, Pixel third);

Pixel findLeftPixel(Component component);

Pixel findRightPixel(Component component);

Pixel findFarthestPixel(std::vector<Pixel>& pixels, LineSegment lineSegment);

std::vector<Pixel> quickHull(Component component);

std::vector<Hull> quickHullAll(std::vector<Component>& components);

std::vector<int> packHulls(std::vector<Hull>& hulls);

std::vector<Hull> unpackHulls(std::vector<int>& packed, int length);

}  // namespace voroshilov_v_convex_hull_components_seq
