/// Register a type as a [residual](crate::residuals) for serialization.
#[macro_export]
macro_rules! tag_residual {
    ($($ty:ty),* $(,)?) => {$(
        $crate::register_typetag!($crate::residuals::ResidualSafe, $ty);
    )*};
}
