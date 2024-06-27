#[macro_export]
macro_rules! impl_residual {
    ($num:expr, $name:ident $(< $T:ident : $Trait:ident >)?, $($vars:ident),* ) => {
        use paste::paste;
        paste!{
            impl$(<$T: $Trait + 'static>)? Residual for $name $(< $T >)?
            {
                type DimOut = <Self as [<Residual $num>]>::DimOut;
                type NumVars = $crate::linalg::Const<$num>;

                fn residual(&self, values: &Values, keys: &[$crate::containers::Symbol]) -> VectorX {
                    self.[<residual $num _single>](values, keys)
                }

                fn residual_jacobian(&self, values: &Values, keys: &[$crate::containers::Symbol]) -> DiffResult<VectorX, MatrixX> {
                    self.[<residual $num _jacobian>](values, keys)
                }
            }
        }
    };
}
