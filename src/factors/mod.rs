mod noise;
pub use noise::GaussianNoise;

use crate::make_enum_robust;

mod robust;
pub use robust::*;
make_enum_robust!(
    RobustEnum,
    L2,
    L1,
    Huber,
    Fair,
    Cauchy,
    GemanMcClure,
    Welsch,
    Tukey
);

mod factor;
pub use factor::Factor;
