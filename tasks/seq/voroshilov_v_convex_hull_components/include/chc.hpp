#pragma once

#include <vector>

namespace voroshilov_v_convex_hull_components_seq {

struct Pixel {
  int y;      // vertical coordinate, 0 is top
  int x;      // horizontal coordinate, 0 is left
  int value;  // the pixel itself

  Pixel() = default;
  Pixel(int y_param, int x_param);
  Pixel(int y_param, int x_param, int value_param);
  Pixel(const Pixel& other);

  Pixel& operator=(const Pixel& other) = default;
  bool operator==(int value_param) const;
  bool operator==(const Pixel& other) const;
  //bool operator!=(const Pixel& other) const;
};

struct Image {
  int height;
  int width;
  std::vector<Pixel> pixels;

  Image() = default;
  Image(int hght, int wdth, std::vector<int> pxls);
  Image(const Image& other);
  Image& operator=(const Image& other) = default;
  Pixel& GetPixel(int y, int x);
};

struct Component {
  std::vector<Pixel> pixels;

  void AddPixel(const Pixel& pixel);
};

struct LineSegment {
  Pixel a;
  Pixel b;

  LineSegment(Pixel& a_param, Pixel& b_param);
};

struct Hull {
  std::vector<Pixel> pixels;

  Hull() = default;
};

Component DepthComponentSearch(Pixel& start_pixel, Image* tmp_image, int index);

std::vector<Component> FindComponents(Image& image);

int CheckRotation(Pixel& first, Pixel& second, Pixel& third);

Pixel FindLeftPixel(Component component);

Pixel FindRightPixel(Component component);

Pixel FindFarthestPixel(std::vector<Pixel>& pixels, LineSegment& line_segment);

std::vector<Pixel> QuickHull(Component component);

std::vector<Hull> QuickHullAll(std::vector<Component>& components);

std::vector<int> PackHulls(std::vector<Hull>& hulls);

std::vector<Hull> UnpackHulls(std::vector<int>& packed, int length);

}  // namespace voroshilov_v_convex_hull_components_seq
