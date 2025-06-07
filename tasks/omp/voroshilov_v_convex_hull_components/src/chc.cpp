#include "../include/chc.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>





#include <chrono>
#include <iostream>




using namespace voroshilov_v_convex_hull_components_omp;

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

std::vector<Component>
voroshilov_v_convex_hull_components_omp::LabelsToComponents(
    std::vector<int>& labels,
    Image& image,
    int /*num_components—больше не нужен*/)
{
    int height = image.height;
    int width  = image.width;
    int N      = height * width;

    // 0) вычисляем реальный максимум метки
    int max_label = 0;
    for (int v : labels) if (v > max_label) max_label = v;

    // если нет никаких компонент
    if (max_label < 2) return {};

    int B = max_label + 1;            // диапазон меток [0..max_label]
    int T = omp_get_max_threads();
    if (T < 1) T = 1;

    // 1) локальные счётчики для каждой метки
    std::vector<std::vector<int>> localCountsThr(T, std::vector<int>(B, 0));

    #pragma omp parallel
    {
      int t = omp_get_thread_num();
      int chunk = (N + T - 1) / T;
      int i0 = t * chunk;
      int i1 = std::min(i0 + chunk, N);
      auto & locCounts = localCountsThr[t];
      for (int i = i0; i < i1; ++i) {
        int lab = labels[i];
        if (lab > 1) locCounts[lab]++;
      }
    }

    // 2) глобальные суммы
    std::vector<int> counts(B, 0);
    for (int lab = 0; lab <= max_label; ++lab) {
      int s = 0;
      for (int t = 0; t < T; ++t) s += localCountsThr[t][lab];
      counts[lab] = s;
    }

    // 3) префикс-сумма
    std::vector<int> startPos(B);
    int total = 0;
    for (int lab = 0; lab <= max_label; ++lab) {
      startPos[lab] = total;
      total += counts[lab];
    }

    // 4) offsets для каждого потока
    std::vector<std::vector<int>> precomputedOffsets(T, std::vector<int>(B,0));
    for (int lab = 0; lab <= max_label; ++lab) {
      int run = startPos[lab];
      for (int t = 0; t < T; ++t) {
        precomputedOffsets[t][lab] = run;
        run += localCountsThr[t][lab];
      }
    }

    // 5) соберём все flat-индексы
    std::vector<int> allIdx(total);
    std::vector<std::vector<int>> localWrittenThr(T, std::vector<int>(B,0));

    #pragma omp parallel
    {
      int t = omp_get_thread_num();
      int chunk = (N + T - 1) / T;
      int i0 = t * chunk;
      int i1 = std::min(i0 + chunk, N);
      auto & written = localWrittenThr[t];
      auto & offsets = precomputedOffsets[t];
      for (int i = i0; i < i1; ++i) {
        int lab = labels[i];
        if (lab > 1) {
          int pos = offsets[lab] + written[lab]++;
          allIdx[pos] = i;
        }
      }
    }

    // 6) финальная сборка компонентов
    int M = max_label - 1; // метки 2..max_label → M = max_label-1 штук
    std::vector<Component> components(M);
    #pragma omp parallel for schedule(dynamic,1)
    for (int L = 2; L <= max_label; ++L) {
      int cnt    = counts[L];
      int offset = startPos[L];
      Component comp;
      comp.reserve(cnt);
      for (int k = 0; k < cnt; ++k) {
        int idx = allIdx[offset + k];
        int y   = idx / width;
        int x   = idx % width;
        comp.emplace_back(y, x, L);
      }
      components[L-2] = std::move(comp);
    }

    return components;
}

std::vector<Component> voroshilov_v_convex_hull_components_omp::FindComponentsOMP(Image& image) {
  int height = image.height;
  int width = image.width;
  int N = height * width;

  int num_threads = omp_get_max_threads();
  if (num_threads < 1) {
    num_threads = 1;
  }

  std::vector<int> start_y(num_threads);
  std::vector<int> end_y(num_threads);
  int base_h = height / num_threads;
  int rem = height % num_threads;
  int cur_y = 0;
  for (int t = 0; t < num_threads; t++) {
    int h = base_h + (t < rem ? 1 : 0);
    start_y[t] = cur_y;
    end_y[t] = cur_y + h;
    cur_y += h;
  }

  std::vector<std::vector<int>> remapLocMap(num_threads);
  std::vector<std::vector<int>> remapLocList(num_threads);
  for(int t = 0; t < num_threads; ++t) {
      int h   = end_y[t] - start_y[t];
      int area= h * width;
      remapLocMap[t].assign(area + 1, 0);      // все нули
      remapLocList[t].reserve(area / 8);
  }

  std::vector<int> labels(N, 0);
  std::vector<int> localCounts(num_threads, 0);
  std::vector<UnionFind> ufs;                             
  ufs.reserve(num_threads);

  for (int t = 0; t < num_threads; t++) {
      int h = end_y[t] - start_y[t];
      int area = h * width;
      ufs.emplace_back(area);
  }

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int y0 = start_y[tid];
    int y1 = end_y[tid];
    int h = y1 - y0;
    int areaSize = h * width;

    // FirstPass:
    auto start = std::chrono::high_resolution_clock::now();

    UnionFind& uf = ufs[tid];
    int offsetRow = y0 * width;

    for (int localIdx = 0; localIdx < areaSize; localIdx++) {
      int globalIdx = offsetRow + localIdx;
      if (image.pixels[globalIdx] == 0) {
        labels[globalIdx] = 0;
        continue;
      }
      labels[globalIdx] = localIdx + 1;

      int y = y0 + (localIdx / width);
      int x = localIdx % width;

      if (x > 0 && image.pixels[globalIdx - 1] == 1) {
        uf.Union(localIdx, localIdx - 1);
      }
      if (y > y0) {
        int aboveLocal = (y - y0 - 1) * width + x;
        if (image.pixels[globalIdx - width] == 1) {
          uf.Union(localIdx, aboveLocal);
        }
        if (x > 0 && image.pixels[globalIdx - width - 1] == 1) {
          uf.Union(localIdx, aboveLocal - 1);
        }
        if (x + 1 < width && image.pixels[globalIdx - width + 1] == 1) {
          uf.Union(localIdx, aboveLocal + 1);
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[FirstPass = " << duration.count() << " ms] \n";

    // Renumerate:
    start = std::chrono::high_resolution_clock::now();

    auto & mapp = remapLocMap[tid];    // вектор length=areaSize+1, изначально 0
    auto & lst  = remapLocList[tid];   // список встреченных корней
    lst.clear();                       // чистим из предыдущего запуска
    int nextLocalLab = 0;

    for(int localIdx = 0; localIdx < areaSize; ++localIdx) {
      int i = offsetRow + localIdx;
      if (image.pixels[i] == 0) {
        labels[i] = 0;
        continue;
      }
      int root = uf.FindRoot(localIdx);
      if (mapp[root] == 0) {
        mapp[root] = ++nextLocalLab;
        lst.push_back(root);
      }
      labels[i] = mapp[root];
    }
    localCounts[tid] = nextLocalLab;

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[Renumerate = " << duration.count() << " ms] \n";
  } // omp parallel
  

  // Offset:
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<int> offsets(num_threads, 0);
  for (int t = 1; t < num_threads; t++) {
      offsets[t] = offsets[t - 1] + localCounts[t - 1];
  }
  int totalBeforeMerge = offsets[num_threads - 1] + localCounts[num_threads - 1];

#pragma omp parallel for schedule(static)
  for (int t = 0; t < num_threads; t++) {
    int y0 = start_y[t];
    int y1 = end_y[t];
    int areaSize = (y1 - y0) * width;
    int offsetRow = y0 * width;
    int offLab = offsets[t];
    for (int localIdx = 0; localIdx < areaSize; localIdx++) {
      int i = offsetRow + localIdx;
      if (labels[i] > 0) {
        labels[i] += offLab;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "[Offset = " << duration.count() << " ms] \n";

  // Merge:
  start = std::chrono::high_resolution_clock::now();

  UnionFind ufGlobal(totalBeforeMerge + 1);

  for (int t = 1; t < num_threads; t++) {
    int yTop = end_y[t-1] - 1;
    int yBottom = start_y[t];
    int globalTop = yTop * width;
    int globalBottom = yBottom * width;
    for (int x = 0; x < width; x++) {
      int labTop = labels[globalTop + x];
      int labBottom = labels[globalBottom + x];
      if (labTop > 0 && labBottom > 0) {
        ufGlobal.Union(labTop, labBottom);
      }
      if (x > 0) {
        int labBottom2 = labels[globalBottom + (x - 1)];
        if (labTop > 0 && labBottom2 > 0) {
          ufGlobal.Union(labTop, labBottom2);
        }
      }
      if (x + 1 < width) {
        int labBottom3 = labels[globalBottom + (x + 1)];
        if (labTop > 0 && labBottom3 > 0) {
          ufGlobal.Union(labTop, labBottom3);
        }
      }
    }
  }

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "[Merge = " << duration.count() << " ms] \n";

  // FinalLabeling:
  start = std::chrono::high_resolution_clock::now();

  std::vector<int> finalMap(totalBeforeMerge + 1, 0);
  std::vector<int> finalList;
  finalList.reserve(totalBeforeMerge);
  int nextFinal = 0;
  for(int lab = 1; lab <= totalBeforeMerge; ++lab) {
    int root = ufGlobal.FindRoot(lab);
    if(finalMap[root] == 0) {
      finalMap[root] = ++nextFinal;
      finalList.push_back(root);
    }
  }

  #pragma omp parallel for schedule(static)
  for(int i = 0; i < N; ++i) {
    int l = labels[i];
    if(l > 0) {
      int r = ufGlobal.FindRoot(l);
      labels[i] = finalMap[r];
    }
  }

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "[FinalLabeling = " << duration.count() << " ms] \n";

  start = std::chrono::high_resolution_clock::now();

  std::vector<Component> final_components = LabelsToComponents(labels, image, nextFinal + 1);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "[LabelsToComponents = " << duration.count() << " ms] \n";

  return final_components;
}

int voroshilov_v_convex_hull_components_omp::CheckRotation(Pixel& first, Pixel& second, Pixel& third) {
  return ((second.x - first.x) * (third.y - second.y)) - ((second.y - first.y) * (third.x - second.x));
}

Pixel voroshilov_v_convex_hull_components_omp::FindFarthestPixel(std::vector<Pixel>& pixels,
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

std::vector<Pixel> voroshilov_v_convex_hull_components_omp::QuickHull(Component& component) {
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

std::vector<Hull> voroshilov_v_convex_hull_components_omp::QuickHullAllOMP(std::vector<Component>& components) {
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

void voroshilov_v_convex_hull_components_omp::PackHulls(std::vector<Hull>& hulls, int width, int height,
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
