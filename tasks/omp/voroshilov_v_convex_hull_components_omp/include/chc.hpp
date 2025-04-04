#pragma once

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace voroshilov_v_convex_hull_components_omp {

struct Pixel {
  int y;      // vertical coordinate, 0 is top
  int x;      // horizontal coordinate, 0 is left
  int value;  // the pixel itself

  Pixel() = default;
  Pixel(int y_param, int x_param);
  NOINLINE Pixel(int y_param, int x_param, int value_param);
  Pixel(const Pixel& other) = default;

  Pixel& operator=(const Pixel& other) = default;
  NOINLINE bool operator==(int value_param) const;
  NOINLINE bool operator==(const Pixel& other) const;
};

struct Image {
  int height;
  int width;
  std::vector<Pixel> pixels;

  Image() = default;
  Image(int hght, int wdth, std::vector<int> pxls);
  Image(const Image& other) = default;
  Image& operator=(const Image& other) = default;
  NOINLINE Pixel& GetPixel(int y, int x);
};

/*struct Component {
  std::vector<Pixel> pixels;
};*/

struct LineSegment {
  Pixel a;
  Pixel b;

  NOINLINE LineSegment(Pixel& a_param, Pixel& b_param);
};

/*struct Hull {
  std::vector<Pixel> pixels;

  Hull() = default;
  Hull(std::vector<Pixel>& pxls);

  bool operator==(const Hull& other) const;
};*/

using Component = std::vector<Pixel>;
using Hull = std::vector<Pixel>;

class UnionFind {
 public:
  std::unordered_map<int, int> roots;
  std::unordered_map<int, int> ranks;

  UnionFind() = default;
  int FindRoot(int x);
  void Union(int x, int y);
};

void CheckBoundaryPixels(UnionFind* union_find, Image& image, int y, int x);

void MergeComponentsAcrossAreas(std::vector<Component>& components, Image& image, int area_height,
                                std::vector<int>& end_y);

Component DepthComponentSearchInArea(Pixel start_pixel, Image* tmp_image, int index, int start_y, int end_y);

std::vector<Component> FindComponentsInArea(Image& tmp_image, int start_y, int end_y, int index_offset);

std::vector<Component> FindComponentsOMP(Image& image);

NOINLINE double CheckRotation(Pixel& first, Pixel& second, Pixel& third);

Pixel FindFarthestPixel(std::vector<Pixel>& pixels, LineSegment& line_segment);

std::vector<Pixel> QuickHull(Component& component);

std::vector<Hull> QuickHullAllOMP(std::vector<Component>& components);

std::pair<std::vector<int>, std::vector<int>> PackHulls(std::vector<Hull>& hulls, Image& image);

std::vector<Hull> UnpackHulls(std::vector<int>& hulls_indexes, std::vector<int>& pixels_indexes, int height, int width,
                              size_t hulls_size);

}  // namespace voroshilov_v_convex_hull_components_omp
