use std::ops::Mul;

use crate::dtype;
use faer::{
    prelude::SpSolver,
    sparse::{linalg::solvers, SparseColMatRef},
    Mat, MatRef,
};

pub trait LinearSolver: Default {
    fn solve_symmetric(&mut self, a: SparseColMatRef<usize, dtype>, b: MatRef<dtype>)
        -> Mat<dtype>;

    fn solve_lst_sq(&mut self, a: SparseColMatRef<usize, dtype>, b: MatRef<dtype>) -> Mat<dtype>;
}

// ------------------------- Cholesky Linear Solver ------------------------- //

#[derive(Default)]
pub struct CholeskySolver {
    sparsity_pattern: Option<solvers::SymbolicCholesky<usize>>,
}

impl LinearSolver for CholeskySolver {
    fn solve_symmetric(
        &mut self,
        a: SparseColMatRef<usize, dtype>,
        b: MatRef<dtype>,
    ) -> Mat<dtype> {
        if self.sparsity_pattern.is_none() {
            self.sparsity_pattern =
                Some(solvers::SymbolicCholesky::try_new(a.symbolic(), faer::Side::Lower).unwrap());
        }

        solvers::Cholesky::try_new_with_symbolic(
            self.sparsity_pattern.clone().unwrap(),
            a,
            faer::Side::Lower,
        )
        .unwrap()
        .solve(&b)
    }

    fn solve_lst_sq(&mut self, a: SparseColMatRef<usize, dtype>, b: MatRef<dtype>) -> Mat<dtype> {
        let ata = a.transpose().to_col_major().unwrap().mul(a);

        let atb = a.transpose().mul(b);

        self.solve_symmetric(ata.as_ref(), atb.as_ref())
    }
}

// ------------------------- QR Linear Solver ------------------------- //

#[derive(Default)]
pub struct QRSolver {
    sparsity_pattern: Option<solvers::SymbolicQr<usize>>,
}

impl LinearSolver for QRSolver {
    fn solve_symmetric(
        &mut self,
        a: SparseColMatRef<usize, dtype>,
        b: MatRef<dtype>,
    ) -> Mat<dtype> {
        self.solve_lst_sq(a, b)
    }

    fn solve_lst_sq(&mut self, a: SparseColMatRef<usize, dtype>, b: MatRef<dtype>) -> Mat<dtype> {
        if self.sparsity_pattern.is_none() {
            self.sparsity_pattern = Some(solvers::SymbolicQr::try_new(a.symbolic()).unwrap());
        }

        // TODO: I think we're doing an extra copy here from solution -> slice solution
        solvers::Qr::try_new_with_symbolic(self.sparsity_pattern.clone().unwrap(), a)
            .unwrap()
            .solve(&b)
            .as_ref()
            .subrows(0, a.ncols())
            .to_owned()
    }
}

// ------------------------- LU Linear Solver ------------------------- //

#[derive(Default)]
pub struct LUSolver {
    sparsity_pattern: Option<solvers::SymbolicLu<usize>>,
}

impl LinearSolver for LUSolver {
    fn solve_symmetric(
        &mut self,
        a: SparseColMatRef<usize, dtype>,
        b: MatRef<dtype>,
    ) -> Mat<dtype> {
        if self.sparsity_pattern.is_none() {
            self.sparsity_pattern = Some(solvers::SymbolicLu::try_new(a.symbolic()).unwrap());
        }

        solvers::Lu::try_new_with_symbolic(self.sparsity_pattern.clone().unwrap(), a.as_ref())
            .unwrap()
            .solve(&b)
    }

    fn solve_lst_sq(&mut self, a: SparseColMatRef<usize, dtype>, b: MatRef<dtype>) -> Mat<dtype> {
        let ata = a.transpose().to_col_major().unwrap().mul(a);
        let atb = a.transpose().mul(b);

        self.solve_symmetric(ata.as_ref(), atb.as_ref())
    }
}

#[cfg(test)]
mod test {
    use faer::{mat, sparse::SparseColMat};
    use matrixcompare::assert_matrix_eq;

    use super::*;

    fn solve<T: LinearSolver>(solver: &mut T) {
        let a = SparseColMat::<usize, dtype>::try_new_from_triplets(
            3,
            2,
            &[
                (0, 0, 10.0),
                (1, 0, 2.0),
                (2, 0, 3.0),
                (0, 1, 4.0),
                (1, 1, 20.0),
                (2, 1, -45.0),
            ],
        )
        .unwrap();
        let b = mat![[15.0], [-3.0], [33.0]];

        let x_exp = mat![[1.874901], [-0.566112]];
        let x = solver.solve_lst_sq(a.as_ref(), b.as_ref());
        println!("{:?}", x);

        assert_matrix_eq!(x, x_exp, comp = abs, tol = 1e-6);
    }

    #[test]
    fn test_cholesky_solver() {
        let mut solver = CholeskySolver::default();
        solve(&mut solver);
    }

    #[test]
    fn test_qr_solver() {
        let mut solver = QRSolver::default();
        solve(&mut solver);
    }

    #[test]
    fn test_lu_solver() {
        let mut solver = LUSolver::default();
        solve(&mut solver);
    }
}
