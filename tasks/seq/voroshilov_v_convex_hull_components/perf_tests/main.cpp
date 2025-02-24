#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

std::vector<int> genBinVec(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> binVec;

  for (int i = 0; i < size; i++) {
    binVec.push_back(gen() % 2);
  }

  return binVec;
}

TEST(voroshilov_v_convex_hull_components_seq, chc_pipeline_run) {
  std::vector<int> pixels = genBinVec(1'000'000);
  int height = 1'000;
  int width = 1'000;

  int* pHeight = &height;
  int* pWidth = &width;
  std::vector<int> out(1'000'000);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pHeight));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pWidth));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  auto chcTaskSequential = std::make_shared<voroshilov_v_convex_hull_components_seq::ChcTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(chcTaskSequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(voroshilov_v_convex_hull_components_seq, chc_task_run) {
  std::vector<int> pixels = genBinVec(1'000'000);
  int height = 1'000;
  int width = 1'000;

  int* pHeight = &height;
  int* pWidth = &width;
  std::vector<int> out(1'000'000);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pHeight));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pWidth));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  auto chcTaskSequential = std::make_shared<voroshilov_v_convex_hull_components_seq::ChcTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(chcTaskSequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
