#[allow(non_camel_case_types)]
pub type dtype = f64;

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
