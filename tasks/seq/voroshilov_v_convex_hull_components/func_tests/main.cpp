#include <gtest/gtest.h>

#include <string>
#include <vector>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

namespace {

bool validationTest(int height, int width, std::vector<int>& pixels) {
  int* pHeight = &height;
  int* pWidth = &width;
  std::vector<int> out(100);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pHeight));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pWidth));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskSequential chcTaskSequential(task_data_seq);
  return chcTaskSequential.ValidationImpl();
}

std::vector<Hull> simpleRunTest(int height, int width, std::vector<int>& pixels) {
  int* pHeight = &height;
  int* pWidth = &width;
  std::vector<int> out(100);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pHeight));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pWidth));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskSequential chcTaskSequential(task_data_seq);
  chcTaskSequential.ValidationImpl();
  chcTaskSequential.PreProcessingImpl();
  chcTaskSequential.RunImpl();
  chcTaskSequential.PostProcessingImpl();

  int outSize = static_cast<int>(task_data_seq->outputs_count[0]);
  std::vector<Hull> hulls = unpackHulls(out, outSize);

  return hulls;
}
#ifndef _WIN32
bool imageRunTest(std::string srcPath, std::string expPath) {
  // Load source image:
  cv::Mat srcImage = cv::imread(srcPath);
  if (srcImage.empty()) {
    return false;
  }

  // Convert to shades of gray:
  cv::Mat grayImage;
  cv::cvtColor(srcImage, grayImage, cv::COLOR_BGR2GRAY);

  // Convert to black and white:
  cv::Mat binImage;
  cv::threshold(grayImage, binImage, 128, 1, cv::THRESH_BINARY);

  // Convert to std::vector<int>:
  std::vector<int> pixels(binImage.rows * binImage.cols);
  for (int i = 0; i < binImage.rows; i++) {
    for (int j = 0; j < binImage.cols; j++) {
      pixels[i * binImage.cols + j] = binImage.at<uchar>(i, j);
    }
  }

  int* pHeight = &binImage.rows;
  int* pWidth = &binImage.cols;
  std::vector<int> out(1000);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pHeight));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pWidth));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskSequential chcTaskSequential(task_data_seq);
  chcTaskSequential.ValidationImpl();
  chcTaskSequential.PreProcessingImpl();
  chcTaskSequential.RunImpl();
  chcTaskSequential.PostProcessingImpl();

  int outSize = static_cast<int>(task_data_seq->outputs_count[0]);
  std::vector<Hull> hulls = unpackHulls(out, outSize);

  // Draw hulls on source image:
  for (Hull hull : hulls) {
    for (size_t i = 0; i < hull.pixels.size() - 1; i++) {
      cv::circle(srcImage, cv::Point(hull.pixels[i].x, hull.pixels[i].y), 2, cv::Scalar(0, 0, 255), cv::FILLED);

      cv::line(srcImage, cv::Point(hull.pixels[i].x, hull.pixels[i].y),
               cv::Point(hull.pixels[i + 1].x, hull.pixels[i + 1].y), cv::Scalar(0, 0, 255), 1);
    }
    cv::circle(srcImage, cv::Point(hull.pixels[hull.pixels.size() - 1].x, hull.pixels[hull.pixels.size() - 1].y), 2,
               cv::Scalar(0, 0, 255), cv::FILLED);

    cv::line(srcImage, cv::Point(hull.pixels[hull.pixels.size() - 1].x, hull.pixels[hull.pixels.size() - 1].y),
             cv::Point(hull.pixels[0].x, hull.pixels[0].y), cv::Scalar(0, 0, 255), 1);
  }

  // Load expected image:
  cv::Mat expImage = cv::imread(expPath);
  if (expImage.empty()) {
    return false;
  }

  // Compare edited source image with expected image:
  double difference = cv::norm(srcImage, expImage);

  // They are same if difference == 0.0
  if (difference > 0.0) {
    return false;
  } else {
    return true;
  }
}
#endif
}  // namespace
TEST(voroshilov_v_convex_hull_components_seq, simpleValidationTest) {
  std::vector<int> pixels = {0, 1, 0, 1, 1, 1, 0, 1, 0};
  int height = 0;
  int width = 3;

  ASSERT_FALSE(validationTest(height, width, pixels));
}

TEST(voroshilov_v_convex_hull_components_seq, simpleTest0Components) {
  std::vector<int> pixels = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int height = 3;
  int width = 3;
  std::vector<Hull> resultHulls = simpleRunTest(height, width, pixels);

  size_t expSize = 0;
  ASSERT_EQ(resultHulls.size(), expSize);
}

TEST(voroshilov_v_convex_hull_components_seq, simpleTest1Component) {
  std::vector<int> pixels = {0, 1, 0, 1, 1, 1, 0, 1, 0};
  int height = 3;
  int width = 3;
  std::vector<Hull> resultHulls = simpleRunTest(height, width, pixels);

  std::vector<Hull> expectHulls;
  Hull hull;
  hull.pixels = {{1, 0}, {0, 1}, {1, 2}, {2, 1}};  // First coordinate is Y, second is X!!!
  expectHulls.push_back(hull);

  ASSERT_EQ(resultHulls.size(), expectHulls.size());

  for (size_t i = 0; i < resultHulls.size(); i++) {
    for (size_t j = 0; j < resultHulls[i].pixels.size(); j++) {
      EXPECT_EQ(resultHulls[i].pixels[j], expectHulls[i].pixels[j]);
    }
  }
}

TEST(voroshilov_v_convex_hull_components_seq, simpleTest3Components) {
  std::vector<int> pixels = {1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0};
  int height = 5;
  int width = 6;
  std::vector<Hull> resultHulls = simpleRunTest(height, width, pixels);

  std::vector<Hull> expectHulls;

  Hull hull1;
  hull1.pixels = {{0, 0}, {0, 1}, {1, 0}};
  expectHulls.push_back(hull1);

  Hull hull2;
  hull2.pixels = {{0, 4}, {0, 5}, {1, 5}, {1, 4}};
  expectHulls.push_back(hull2);

  Hull hull3;
  hull3.pixels = {{4, 0}, {2, 2}, {3, 3}, {4, 1}};
  expectHulls.push_back(hull3);

  ASSERT_EQ(resultHulls.size(), expectHulls.size());

  for (size_t i = 0; i < resultHulls.size(); i++) {
    for (size_t j = 0; j < resultHulls[i].pixels.size(); j++) {
      EXPECT_EQ(resultHulls[i].pixels[j], expectHulls[i].pixels[j]);
    }
  }
}

TEST(voroshilov_v_convex_hull_components_seq, simpleTest5Components) {
  std::vector<int> pixels = {1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                             0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
  int height = 11;
  int width = 10;
  std::vector<Hull> resultHulls = simpleRunTest(height, width, pixels);

  std::vector<Hull> expectHulls;

  Hull hull1;
  hull1.pixels = {{0, 0}, {0, 1}, {1, 2}, {2, 2}, {2, 1}, {1, 0}};
  expectHulls.push_back(hull1);

  Hull hull2;
  hull2.pixels = {{1, 4}, {0, 5}, {0, 6}, {1, 7}, {2, 6}, {2, 5}};
  expectHulls.push_back(hull2);

  Hull hull3;
  hull3.pixels = {{8, 7}, {5, 8}, {6, 9}, {10, 9}};
  expectHulls.push_back(hull3);

  Hull hull4;
  hull4.pixels = {{7, 1}, {6, 2}, {7, 3}, {8, 2}};
  expectHulls.push_back(hull4);

  Hull hull5;
  hull5.pixels = {{9, 4}, {8, 5}, {10, 5}};
  expectHulls.push_back(hull5);

  ASSERT_EQ(resultHulls.size(), expectHulls.size());

  for (size_t i = 0; i < resultHulls.size(); i++) {
    for (size_t j = 0; j < resultHulls[i].pixels.size(); j++) {
      EXPECT_EQ(resultHulls[i].pixels[j], expectHulls[i].pixels[j]);
    }
  }
}
#ifndef _WIN32
TEST(voroshilov_v_convex_hull_components_seq, imageTest0) {
  std::string srcPath = "../data/0_image.png";
  std::string expPath = "../data/0_expected.png";

  ASSERT_TRUE(imageRunTest(srcPath, expPath));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest0Incorrect) {
  std::string srcPath = "../data/0_image.png";
  std::string expPath = "../data/0_incorrect.png";

  ASSERT_FALSE(imageRunTest(srcPath, expPath));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest1) {
  std::string srcPath = "../data/1_image.png";
  std::string expPath = "../data/1_expected.png";

  ASSERT_TRUE(imageRunTest(srcPath, expPath));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest2) {
  std::string srcPath = "../data/2_image.png";
  std::string expPath = "../data/2_expected.png";

  ASSERT_TRUE(imageRunTest(srcPath, expPath));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest3) {
  std::string srcPath = "../data/3_image.png";
  std::string expPath = "../data/3_expected.png";

  ASSERT_TRUE(imageRunTest(srcPath, expPath));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest4) {
  std::string srcPath = "../data/4_image.png";
  std::string expPath = "../data/4_expected.png";

  ASSERT_TRUE(imageRunTest(srcPath, expPath));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest5) {
  std::string srcPath = "../data/5_image.png";
  std::string expPath = "../data/5_expected.png";

  ASSERT_TRUE(imageRunTest(srcPath, expPath));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest6) {
  std::string srcPath = "../data/6_image.png";
  std::string expPath = "../data/6_expected.png";

  ASSERT_TRUE(imageRunTest(srcPath, expPath));
}
#endif