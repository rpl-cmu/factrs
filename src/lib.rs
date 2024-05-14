pub mod factors;
pub mod macros;
pub mod traits;
pub mod variables;
struct DefaultBundle;

impl traits::Bundle for DefaultBundle {
    type Key = variables::Symbol;
    type Variable = variables::VariableEnum;
    type Robust = (); // TODO
    type Noise = (); // TODO
    type Residual = (); // TODO
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
