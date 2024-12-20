#include "gtsam/slam/dataset.h"
#include <chrono>
#include <cstdio>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <type_traits>

using namespace gtsam;

GraphAndValues load(std::string file, bool is3D) {
  auto read = readG2o(file, is3D);

  if (is3D) {
    auto priorModel = noiseModel::Diagonal::Variances(
        (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    read.first->addPrior(0, Pose3::Identity(), priorModel);
  } else {
    auto priorModel =
        noiseModel::Diagonal::Variances(Vector3(1e-6, 1e-6, 1e-8));
    auto prior = PriorFactor<Pose2>(0, Pose2());
    read.first->addPrior(0, Pose2::Identity(), priorModel);
  }

  return read;
}

void run(const NonlinearFactorGraph &const_graph, const Values &const_values) {
  NonlinearFactorGraph graph(const_graph);
  Values values(const_values);

  GaussNewtonOptimizer optimizer(graph, values);
  Values result = optimizer.optimize();
  // Put something in to force it from being optimized out
  result.update(0, Pose3::Identity());
}

std::string directory = "../../../examples/data/";
std::vector<std::string> files_3d{"sphere2500.g2o", "parking-garage.g2o"};
std::vector<std::string> files_2d{"M3500.g2o"};
// std::vector<std::tuple(std::string, std::is_function<typename>)

int main(int argc, char *argv[]) {
  size_t sample_count;
  if (argc > 1) {
    sample_count = std::atoi(argv[1]);
  } else {
    sample_count = 100;
  }

  std::cout << "Beginning 3d trials" << std::endl;
  for (auto &file : files_3d) {
    std::string path = directory + file;
    auto gv = load(path, true);
    std::chrono::duration<double, std::nano> total_time =
        std::chrono::nanoseconds(0);
    for (size_t i = 0; i < sample_count; ++i) {
      auto begin = std::chrono::steady_clock::now();
      run(*gv.first, *gv.second);
      auto end = std::chrono::steady_clock::now();
      total_time += end - begin;
    }
  }
}