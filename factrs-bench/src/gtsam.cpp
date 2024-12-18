#include "factrs-bench/include/gtsam.h"

std::shared_ptr<GraphValues> load_g2o(const std::string &file, bool is3D) {
  auto read = gtsam::readG2o(file, is3D);
  auto graph = gtsam::NonlinearFactorGraph(*read.first);
  auto values = gtsam::Values(*read.second);

  if (is3D) {
    auto priorModel = gtsam::noiseModel::Diagonal::Variances(
        (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(), priorModel));
  } else {
    auto priorModel = gtsam::noiseModel::Diagonal::Variances(
        gtsam::Vector3(1e-6, 1e-6, 1e-8));
    graph.add(gtsam::PriorFactor<gtsam::Pose2>(0, gtsam::Pose2(), priorModel));
  }

  return std::make_shared<GraphValues>(graph, values);
}

void run(const std::shared_ptr<GraphValues> &gv) {
  gtsam::NonlinearFactorGraph graph(gv->graph);
  gtsam::Values values(gv->values);

  gtsam::GaussNewtonOptimizer optimizer(graph, values);
  gtsam::Values result = optimizer.optimize();
}

void hello() { std::cout << "Hello, GTSAM!" << std::endl; }