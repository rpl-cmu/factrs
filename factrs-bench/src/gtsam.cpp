#include "factrs-bench/include/gtsam.h"

#include <iostream>

void gtsam_hello() {
  gtsam::Pose2 x(1.0, 2.0, 3.0);
  std::cout << "Pose2\n" << x << std::endl;
}