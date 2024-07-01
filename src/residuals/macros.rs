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
                where
                    <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<dtype, Self::DimIn>>::Buffer: Sync + Send,
                    nalgebra::DefaultAllocator: $crate::linalg::DualAllocator<Self::DimIn>,
                {
                    self.[<residual $num _jacobian>](values, keys)
                }
            }
        }
    };
}
