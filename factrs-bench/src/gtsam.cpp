#include "factrs-bench/include/gtsam.h"

std::unique_ptr<GraphValues> load_g2o(const std::string &file, bool is3D) {
  auto read = gtsam::readG2o(file, is3D);
  auto out = std::make_unique<GraphValues>(*read.first, *read.second);

  // std::cout << "Adding extra factors" << std::endl;
  // std::cout << "Graph size = " << out->graph.size() << std::endl;

  gtsam::NonlinearFactorGraph graph;
  if (is3D) {
    // auto priorModel = gtsam::noiseModel::Diagonal::Variances(
    //     (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    auto prior = gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3::Identity());
    graph.addPrior(0, gtsam::Pose3::Identity(), gtsam::Matrix6::Identity());
  } else {
    // auto priorModel = gtsam::noiseModel::Diagonal::Variances(
    //     gtsam::Vector3(1e-6, 1e-6, 1e-8));
    auto prior = gtsam::PriorFactor<gtsam::Pose2>(0, gtsam::Pose2());
    graph.add(prior);
  }

  // graph.print();
  // std::cout << "Graph size = " << graph.size() << std::endl;
  // std::cout << "Initial error = " << graph.error(out->values) << std::endl;

  return out;
}

void run(const GraphValues &gv) {
  gtsam::NonlinearFactorGraph graph(gv.graph);
  gtsam::Values values(gv.values);

  gtsam::GaussNewtonOptimizer optimizer(graph, values);
  gtsam::Values result = optimizer.optimize();
}

void hello() { std::cout << "Hello, GTSAM!" << std::endl; }