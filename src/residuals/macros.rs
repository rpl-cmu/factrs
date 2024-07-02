#[macro_export]
macro_rules! impl_residual {
    ($num:expr, $name:ident $(< $T:ident : $Trait:ident >)?, $($vars:ident),* ) => {
        use paste::paste;
        paste!{
            impl$(<$T: $Trait<Alias<dtype> = $T> + 'static>)? Residual for $name $(< $T >)?
            {
                type DimIn = <Self as [<Residual $num>]>::DimIn;
                type DimOut = <Self as [<Residual $num>]>::DimOut;
                type NumVars = $crate::linalg::Const<$num>;

                fn residual(&self, values: &Values, keys: &[$crate::containers::Symbol]) -> VectorX {
                    self.[<residual $num _values>](values, keys)
                }

                fn residual_jacobian(&self, values: &Values, keys: &[$crate::containers::Symbol]) -> DiffResult<VectorX, MatrixX>
                {
                    self.[<residual $num _jacobian>](values, keys)
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_safe_residual {
    ($($var:ident < $T:ident > ),* $(,)?) => {
        use paste::paste;
        $(
            paste!{
                type [<$var $T>] = $var< $T >;
                #[cfg_attr(feature = "serde", typetag::serde)]
                impl $crate::residuals::ResidualSafe for [<$var $T>] {
                    fn dim_in(&self) -> usize {
                        $crate::residuals::Residual::dim_in(self)
                    }

                    fn dim_out(&self) -> usize {
                        $crate::residuals::Residual::dim_out(self)
                    }

                    fn residual(&self, values: &$crate::containers::Values, keys: &[Symbol]) -> VectorX {
                        $crate::residuals::Residual::residual(self, values, keys)
                    }

                    fn residual_jacobian(&self, values: &$crate::containers::Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX> {
                        $crate::residuals::Residual::residual_jacobian(self, values, keys)
                    }
                }
            }
        )*
    };
}
