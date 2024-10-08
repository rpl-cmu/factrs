// TODO: Test on custom residual
// TODO: Make this a procedural macro??
/// Macro to implement the [Residual](crate::residuals::Residual) trait.
///
/// Call this macro after implementing one of the numbered residual traits.
/// Currently works with a single generic on the struct in needed.
#[macro_export]
macro_rules! impl_residual {
    ($num:expr, $name:ident $(< $T:ident : $Trait:ident >)? ) => {
        use paste::paste;
        paste!{
            impl$(<$T: $Trait + 'static>)? $crate::residuals::Residual for $name $(< $T >)?
            {
                type DimIn = <Self as [<Residual $num>]>::DimIn;
                type DimOut = <Self as [<Residual $num>]>::DimOut;
                type NumVars = $crate::linalg::Const<$num>;

                fn residual(&self, values: &$crate::containers::Values, keys: &[$crate::containers::Key]) -> $crate::linalg::VectorX {
                    self.[<residual $num _values>](values, keys)
                }

                fn residual_jacobian(&self, values: &$crate::containers::Values, keys: &[$crate::containers::Key]) -> $crate::linalg::DiffResult<$crate::linalg::VectorX, $crate::linalg::MatrixX>
                {
                    self.[<residual $num _jacobian>](values, keys)
                }
            }
        }
    };
}

/// Register a type as a [residual](crate::residuals) for serialization.
#[macro_export]
macro_rules! tag_residual {
    ($($ty:ty),* $(,)?) => {$(
        $crate::register_typetag!($crate::residuals::ResidualSafe, $ty);
    )*};
}
