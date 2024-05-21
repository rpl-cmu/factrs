// Implements right plus and minus
// TODO: Eventually make a crate feature to switch to right instead
#[macro_export]
macro_rules! liegroup_operators {
    () => {
        fn minus(&self, other: &Self) -> Self {
            &other.inverse() * self
        }

        fn plus(&self, other: &Self) -> Self {
            self * other
        }

        fn oplus(&self, delta: &VectorX<D>) -> Self {
            self.plus(&Self::exp(delta))
        }

        fn ominus(&self, other: &Self) -> VectorX<D> {
            self.minus(other).log()
        }
    };
}
