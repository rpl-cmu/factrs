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

    fn residual_jacobian_numerical(&self, v: &[V]) -> (VectorX<dtype>, MatrixX<dtype>) {
        let eps = 1e-6;
        let dim = v.iter().map(|x| x.dim()).sum();
        let duals: Vec<V::Dual> = v
            .iter()
            .scan(0, |idx, x| {
                let d = x.dual(*idx, dim);
                *idx += x.dim();
                Some(d)
            })
            .collect();

        let fx: VectorX<DualVec> = self.residual(&duals);
        let mut jac: MatrixX<dtype> = MatrixX::zeros(Self::DIM, dim);

        let mut curr_dim = 0;
        for i in 0..v.len() {
            for j in 0..v[i].dim() {
                let mut v_plus = duals.clone();
                let mut tv = v_plus[i].dual_tangent(0, dim);
                tv[j] = DualVec::from(eps);

                v_plus[i] = v_plus[i].oplus(&tv);

                let fx_plus = self.residual(&v_plus);
                let delta: Vec<_> = fx_plus
                    .iter()
                    .zip(fx.iter())
                    .map(|(a, b)| (a.re - b.re) / eps)
                    .collect();
                let delta = VectorX::from(delta);

                jac.columns_mut(curr_dim, 1).copy_from(&delta);
                curr_dim += 1;
            }
        }

        (fx.map(|r| r.re), jac)
    }
}
