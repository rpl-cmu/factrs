use crate::{
    dtype,
    linalg::{
        AllocatorBuffer, Const, DefaultAllocator, DimName, DualAllocator, DualVector, Numeric,
        Vector, VectorDim, VectorViewX, VectorX,
    },
    variables::Variable,
};

// ------------------------- Our needs ------------------------- //
impl<const N: usize, D: Numeric> Variable<D> for Vector<N, D> {
    type Dim = Const<N>;
    type Alias<DD: Numeric> = Vector<N, DD>;

    fn identity() -> Self {
        Vector::zeros()
    }

    fn inverse(&self) -> Self {
        -self
    }

    fn compose(&self, other: &Self) -> Self {
        self + other
    }

    fn exp(delta: VectorViewX<D>) -> Self {
        Self::from_iterator(delta.iter().cloned())
    }

    fn log(&self) -> VectorX<D> {
        VectorX::from_iterator(Self::DIM, self.iter().cloned())
    }

    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD> {
        other.map(|x| x.into())
    }

    // Mostly unncessary, but avoids having to convert VectorX to static vector
    fn dual_setup<NN: DimName>(idx: usize) -> Self::Alias<DualVector<NN>>
    where
        AllocatorBuffer<NN>: Sync + Send,
        DefaultAllocator: DualAllocator<NN>,
        DualVector<NN>: Copy,
    {
        let n = VectorDim::<NN>::zeros().shape_generic().0;
        let mut tv = Self::Alias::<DualVector<NN>>::zeros();
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps = num_dual::Derivative::derivative_generic(n, Const::<1>, idx + i);
        }
        tv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{linalg::Vector6, test_variable};

    // Be lazy and only test Vector6 - others should work the same
    test_variable!(Vector6);
}
