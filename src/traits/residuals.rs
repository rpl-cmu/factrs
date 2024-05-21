use crate::dtype;
use crate::linalg::{Const, Dyn, MatrixX, VectorX};
use crate::traits::{DualVec, Variable};

// TODO: generic on bundle instead??
pub trait Residual<V: Variable<dtype>>: Sized {
    const DIM: usize;

    fn dim(&self) -> usize {
        Self::DIM
    }

    // TODO: Would be nice if this was generic over dtypes, but it'll probably mostly be used with dual vecs
    fn residual(&self, v: &[V::Dual]) -> VectorX<DualVec>;

    fn residual_jacobian(&self, v: &[V]) -> (VectorX<dtype>, MatrixX<dtype>) {
        let dim = v.iter().map(|x| x.dim()).sum();
        let duals: Vec<V::Dual> = v
            .iter()
            .scan(0, |idx, x| {
                let d = x.dual(*idx, dim);
                *idx += x.dim();
                Some(d)
            })
            .collect();

        let res: VectorX<DualVec> = self.residual(&duals);

        let eps = MatrixX::from_rows(
            res.map(|r| r.eps.unwrap_generic(Dyn(dim), Const::<1>).transpose())
                .as_slice(),
        );

        (res.map(|r| r.re), eps)
    }
}
