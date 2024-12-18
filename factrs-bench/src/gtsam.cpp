#include "factrs-bench/include/gtsam.h"

std::shared_ptr<GraphValues> load_g2o(std::string &file, bool is3D) {
  auto read = gtsam::readG2o(file, is3D);
  return std::make_shared<GraphValues>(*read.first, *read.second);
}

void run(const std::shared_ptr<GraphValues> &gv) {
  gtsam::NonlinearFactorGraph graph(gv->graph);
  gtsam::Values values(gv->values);

  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values);
  gtsam::Values result = optimizer.optimize();
}