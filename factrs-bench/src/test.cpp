#include "factrs-bench/include/gtsam.h"

int main() {
  auto gv = load_g2o("../../examples/data/sphere2500.g2o", true);
  std::cout << gv->graph.size() << std::endl;
  run(*gv);
  hello();
  return 0;
}