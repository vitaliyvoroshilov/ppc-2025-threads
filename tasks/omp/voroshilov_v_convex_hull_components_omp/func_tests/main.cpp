#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "omp/voroshilov_v_convex_hull_components_omp/include/chc.hpp"
#include "omp/voroshilov_v_convex_hull_components_omp/include/chc_omp.hpp"

using namespace voroshilov_v_convex_hull_components_omp;

namespace {

void SortHulls(std::vector<Hull>& hulls) {
  std::ranges::sort(hulls, [](const Hull& a, const Hull& b) {
    const Pixel& left_top_a = *std::ranges::min_element(
        a.pixels, [](const Pixel& p1, const Pixel& p2) { return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y); });
    const Pixel& left_top_b = *std::ranges::min_element(
        b.pixels, [](const Pixel& p1, const Pixel& p2) { return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y); });

    return left_top_a.x < left_top_b.x || (left_top_a.x == left_top_b.x && left_top_a.y < left_top_b.y);
  });
}

bool ValidationTest(int height, int width, std::vector<int>& pixels) {
  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskOMP chc_task_omp(task_data_seq);
  return chc_task_omp.ValidationImpl();
}

std::vector<Hull> SimpleRunTest(int height, int width, std::vector<int>& pixels) {
  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskOMP chc_task_omp(task_data_seq);
  chc_task_omp.ValidationImpl();
  chc_task_omp.PreProcessingImpl();
  chc_task_omp.RunImpl();
  chc_task_omp.PostProcessingImpl();

  int hulls_size = static_cast<int>(task_data_seq->outputs_count[0]);
  std::vector<Hull> hulls = UnpackHulls(hulls_indexes_out, pixels_indexes_out, height, width, hulls_size);

  return hulls;
}

#ifndef _WIN32

bool ImageRunTest(std::string& src_path, std::string& exp_path) {
  // Load source image:
  cv::Mat src_image = cv::imread(src_path);
  if (src_image.empty()) {
    return false;
  }

  // Convert to shades of gray:
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, cv::COLOR_BGR2GRAY);

  // Convert to black and white:
  cv::Mat bin_image;
  cv::threshold(gray_image, bin_image, 128, 1, cv::THRESH_BINARY);

  // Convert to std::vector<int>:
  std::vector<int> pixels(bin_image.rows * bin_image.cols);
  for (int i = 0; i < bin_image.rows; i++) {
    for (int j = 0; j < bin_image.cols; j++) {
      pixels[(i * bin_image.cols) + j] = bin_image.at<uchar>(i, j);
    }
  }

  int* p_height = &bin_image.rows;
  int height = bin_image.rows;
  int* p_width = &bin_image.cols;
  int width = bin_image.cols;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskOMP chc_task_omp(task_data_seq);
  chc_task_omp.ValidationImpl();
  chc_task_omp.PreProcessingImpl();
  chc_task_omp.RunImpl();
  chc_task_omp.PostProcessingImpl();

  int hulls_size = static_cast<int>(task_data_seq->outputs_count[0]);
  std::vector<Hull> hulls = UnpackHulls(hulls_indexes_out, pixels_indexes_out, height, width, hulls_size);

  // Draw hulls on source image:
  for (Hull hull : hulls) {
    for (size_t i = 0; i < hull.pixels.size() - 1; i++) {
      cv::circle(src_image, cv::Point(hull.pixels[i].x, hull.pixels[i].y), 2, cv::Scalar(0, 0, 255), cv::FILLED);

      cv::line(src_image, cv::Point(hull.pixels[i].x, hull.pixels[i].y),
               cv::Point(hull.pixels[i + 1].x, hull.pixels[i + 1].y), cv::Scalar(0, 0, 255), 1);
    }
    cv::circle(src_image, cv::Point(hull.pixels[hull.pixels.size() - 1].x, hull.pixels[hull.pixels.size() - 1].y), 2,
               cv::Scalar(0, 0, 255), cv::FILLED);

    cv::line(src_image, cv::Point(hull.pixels[hull.pixels.size() - 1].x, hull.pixels[hull.pixels.size() - 1].y),
             cv::Point(hull.pixels[0].x, hull.pixels[0].y), cv::Scalar(0, 0, 255), 1);
  }

  // Load expected image:
  cv::Mat exp_image = cv::imread(exp_path);
  if (exp_image.empty()) {
    src_image.release();
    gray_image.release();
    bin_image.release();
    return false;
  }

  // cv::imwrite(exp_path, src_image);

  // Compare edited source image with expected image:
  double difference = cv::norm(src_image, exp_image);

  src_image.release();
  gray_image.release();
  bin_image.release();
  exp_image.release();
  // They are same if difference == 0.0
  return difference <= 0.0;
}

#endif

}  // namespace

TEST(voroshilov_v_convex_hull_components_omp, ComponentTest) {
  std::vector<Pixel> pixels = {{0, 0}, {1, 1}, {2, 2}};

  Component component(pixels);

  ASSERT_EQ(component.pixels.size(), pixels.size());

  for (size_t i = 0; i < pixels.size(); i++) {
    EXPECT_EQ(component.pixels[i].y, pixels[i].y);
    EXPECT_EQ(component.pixels[i].x, pixels[i].x);
  }
}

TEST(voroshilov_v_convex_hull_components_omp, HullTest) {
  std::vector<Pixel> pixels1 = {{0, 0}, {1, 1}, {2, 2}};
  std::vector<Pixel> pixels2 = {{0, 0}, {1, 1}, {2, 2}};
  std::vector<Pixel> pixels3 = {{0, 0}, {1, 1}, {3, 3}};

  Hull hull1(pixels1);
  Hull hull2(pixels2);
  Hull hull3(pixels3);

  ASSERT_TRUE(hull1 == hull2);

  ASSERT_FALSE(hull1 == hull3);
}

TEST(voroshilov_v_convex_hull_components_omp, simpleValidationTest) {
  // clang-format off
  std::vector<int> pixels = 
  {0, 1, 0,
   1, 1, 1,
   0, 1, 0};
  // clang-format on
  int height = 0;
  int width = 3;

  ASSERT_FALSE(ValidationTest(height, width, pixels));
}

TEST(voroshilov_v_convex_hull_components_omp, simpleTest0Components) {
  // clang-format off
  std::vector<int> pixels =
  {0, 0, 0,
   0, 0, 0,
   0, 0, 0};
  // clang-format on
  int height = 3;
  int width = 3;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  ASSERT_TRUE(result_hulls.empty());
}

TEST(voroshilov_v_convex_hull_components_omp, simpleTest1Component) {
  // clang-format off
  std::vector<int> pixels =
  {0, 1, 0,
   1, 1, 1,
   0, 1, 0};
  // clang-format on
  int height = 3;
  int width = 3;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;
  std::vector<Pixel> hull_pixels = {{1, 0}, {0, 1}, {1, 2}, {2, 1}};  // First coordinate is Y, second is X!!!
  Hull hull(hull_pixels);
  expect_hulls.push_back(hull);

  ASSERT_EQ(result_hulls.size(), expect_hulls.size());

  for (size_t i = 0; i < result_hulls.size(); i++) {
    EXPECT_EQ(result_hulls[i], expect_hulls[i]);
  }
}

TEST(voroshilov_v_convex_hull_components_omp, simpleTest3Components) {
  // clang-format off
  std::vector<int> pixels = 
  {1, 1, 0, 0, 1, 1,
   1, 0, 0, 0, 1, 1,
   0, 0, 1, 0, 0, 0,
   0, 1, 1, 1, 0, 0,
   1, 1, 0, 0, 0, 0};
  // clang-format on
  int height = 5;
  int width = 6;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;

  std::vector<Pixel> hull1_pixels = {{0, 0}, {0, 1}, {1, 0}};
  Hull hull1(hull1_pixels);
  expect_hulls.push_back(hull1);

  std::vector<Pixel> hull2_pixels = {{0, 4}, {0, 5}, {1, 5}, {1, 4}};
  Hull hull2(hull2_pixels);
  expect_hulls.push_back(hull2);

  std::vector<Pixel> hull3_pixels = {{4, 0}, {2, 2}, {3, 3}, {4, 1}};
  Hull hull3(hull3_pixels);
  expect_hulls.push_back(hull3);

  SortHulls(result_hulls);
  SortHulls(expect_hulls);

  ASSERT_EQ(result_hulls.size(), expect_hulls.size());

  for (size_t i = 0; i < result_hulls.size(); i++) {
    EXPECT_EQ(result_hulls[i], expect_hulls[i]);
  }
}

TEST(voroshilov_v_convex_hull_components_omp, simpleTest5Components) {
  // clang-format off
  std::vector<int> pixels = 
  {1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
   1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
   0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
   0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
   0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
   0, 0, 1, 0, 0, 1, 0, 1, 1, 0,
   0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
   0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
  // clang-format on
  int height = 11;
  int width = 10;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;

  std::vector<Pixel> hull1_pixels = {{0, 0}, {0, 1}, {1, 2}, {2, 2}, {2, 1}, {1, 0}};
  Hull hull1(hull1_pixels);
  expect_hulls.push_back(hull1);

  std::vector<Pixel> hull2_pixels = {{1, 4}, {0, 5}, {0, 6}, {1, 7}, {2, 6}, {2, 5}};
  Hull hull2(hull2_pixels);
  expect_hulls.push_back(hull2);

  std::vector<Pixel> hull3_pixels = {{8, 7}, {5, 8}, {6, 9}, {10, 9}};
  Hull hull3(hull3_pixels);
  expect_hulls.push_back(hull3);

  std::vector<Pixel> hull4_pixels = {{7, 1}, {6, 2}, {7, 3}, {8, 2}};
  Hull hull4(hull4_pixels);
  expect_hulls.push_back(hull4);

  std::vector<Pixel> hull5_pixels = {{9, 4}, {8, 5}, {10, 5}};
  Hull hull5(hull5_pixels);
  expect_hulls.push_back(hull5);

  SortHulls(result_hulls);
  SortHulls(expect_hulls);

  ASSERT_EQ(result_hulls.size(), expect_hulls.size());

  for (size_t i = 0; i < result_hulls.size(); i++) {
    EXPECT_EQ(result_hulls[i], expect_hulls[i]);
  }
}

#ifndef _WIN32

TEST(voroshilov_v_convex_hull_components_omp, imageTest0) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/0_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/0_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest0Incorrect) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/0_image.png");
  std::string inc_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/0_incorrect.png");

  ASSERT_FALSE(ImageRunTest(src_path, inc_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest1) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/1_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/1_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest1Incorrect) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/1_image.png");
  std::string inc_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/1_incorrect.png");

  ASSERT_FALSE(ImageRunTest(src_path, inc_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest2) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/2_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/2_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest3) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/3_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/3_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest4) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/4_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/4_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest5) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/5_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/5_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest6) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/6_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/6_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_omp, imageTest7) {
  std::string src_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/7_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("omp/voroshilov_v_convex_hull_components_omp/data/7_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

#endif
