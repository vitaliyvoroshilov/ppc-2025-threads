#pragma once

#include <vector>

namespace voroshilov_v_convex_hull_components_seq {

struct Pixel {
  int y;      // vertical coordinate, 0 is top
  int x;      // horizontal coordinate, 0 is left
  int value;  // the pixel itself

  Pixel() = default;
  Pixel(int yParam, int xParam);
  Pixel(int yParam, int xParam, int valueParam);
  Pixel(const Pixel& other);
  Pixel& operator=(const Pixel& other) = default;
  Pixel& operator=(int valueParam);
  bool operator==(int valueParam) const;
  bool operator==(const Pixel& other) const;
  bool operator!=(const Pixel& other) const;
};

struct Image {
  int height;
  int width;
  std::vector<Pixel> pixels;

  Image() = default;
  Image(int hght, int wdth, std::vector<int> pxls);
  Image(const Image& other);
  Image& operator=(const Image& other) = default;
  Pixel& getPixel(int y, int x);
};

struct Component {
  std::vector<Pixel> pixels;

  void addPixel(const Pixel& pixel);
};

struct LineSegment {
  Pixel a;
  Pixel b;

  LineSegment(Pixel aParam, Pixel bParam);
};

struct Hull {
  std::vector<Pixel> pixels;

  Hull() = default;
};

Component depthComponentSearch(Pixel& startPixel, Image* tmpImage, int index);

std::vector<Component> findComponents(Image& image);

int checkRotation(Pixel& first, Pixel& second, Pixel& third);

Pixel findLeftPixel(Component component);

Pixel findRightPixel(Component component);

Pixel findFarthestPixel(std::vector<Pixel>& pixels, LineSegment lineSegment);

std::vector<Pixel> quickHull(Component component);

std::vector<Hull> quickHullAll(std::vector<Component>& components);

std::vector<int> packHulls(std::vector<Hull>& hulls);

std::vector<Hull> unpackHulls(std::vector<int>& packed, int length);

}  // namespace voroshilov_v_convex_hull_components_seq
