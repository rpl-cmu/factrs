type dtype = num_dual::DualDVec64;

pub mod factors;
pub mod macros;
pub mod traits;
pub mod variables;
struct DefaultBundle;

impl traits::Bundle for DefaultBundle {
    type Key = variables::Symbol;
    type Variable = variables::VariableEnum;
    type Robust = (); // TODO
    type Noise = factors::NoiseModelEnum;
    type Residual = (); // TODO
}
