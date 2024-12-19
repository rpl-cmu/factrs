#include "gtsam/geometry/Pose2.h"
#include "gtsam/nonlinear/GaussNewtonOptimizer.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"
#include "gtsam/slam/dataset.h"

struct GraphValues {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;

  GraphValues(gtsam::NonlinearFactorGraph graph, gtsam::Values values)
      : graph(graph), values(values) {}
};

std::unique_ptr<GraphValues> load_g2o(const std::string &file, bool is3D);

void run(const GraphValues &gv);

void hello();