use crate::{
    linalg::{Const, DualNum, DualVec, Dyn, Vector, VectorViewX, VectorX},
    variables::Variable,
};

// ------------------------- Our needs ------------------------- //
impl<const N: usize, D: DualNum> Variable<D> for Vector<N, D> {
    type Dim = Const<N>;
    type Dual = Vector<N, DualVec>;

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

    fn dual_self(&self) -> Self::Dual {
        self.map(|x| x.into())
    }

    // Mostly unncessary, but avoids having to convert VectorX to static vector
    fn dual_setup(idx: usize, total: usize) -> Self::Dual {
        let mut tv: Vector<N, DualVec> = Self::Dual::zeros();
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps = num_dual::Derivative::derivative_generic(Dyn(total), Const::<1>, idx + i);
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
