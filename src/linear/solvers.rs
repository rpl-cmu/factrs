use std::ops::Mul;

use crate::dtype;
use faer::prelude::SpSolver;
use faer::sparse::linalg::solvers;
use faer::sparse::SparseColMat;
use faer::Mat;

pub trait LinearSolver: Default {
    fn solve(&mut self, a: &SparseColMat<usize, dtype>, b: &Mat<dtype>) -> Mat<dtype>;
}

// ------------------------- Cholesky Linear Solver ------------------------- //

#[derive(Default)]
pub struct CholeskySolver {
    sparsity_pattern: Option<solvers::SymbolicCholesky<usize>>,
}

impl LinearSolver for CholeskySolver {
    fn solve(&mut self, a: &SparseColMat<usize, dtype>, b: &Mat<dtype>) -> Mat<dtype> {
        let ata = a
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(a.as_ref());

        let atb = a.as_ref().transpose().mul(b);

        if self.sparsity_pattern.is_none() {
            self.sparsity_pattern = Some(
                solvers::SymbolicCholesky::try_new(ata.symbolic(), faer::Side::Lower).unwrap(),
            );
        }

        solvers::Cholesky::try_new_with_symbolic(
            self.sparsity_pattern.clone().unwrap(),
            ata.as_ref(),
            faer::Side::Lower,
        )
        .unwrap()
        .solve(&atb)
    }
}

// ------------------------- QR Linear Solver ------------------------- //

#[derive(Default)]
pub struct QRSolver {
    sparsity_pattern: Option<solvers::SymbolicQr<usize>>,
}

impl LinearSolver for QRSolver {
    fn solve(&mut self, a: &SparseColMat<usize, dtype>, b: &Mat<dtype>) -> Mat<dtype> {
        if self.sparsity_pattern.is_none() {
            self.sparsity_pattern = Some(solvers::SymbolicQr::try_new(a.symbolic()).unwrap());
        }

        // TODO: I think we're doing an extra copy here from solution -> slice solution
        solvers::Qr::try_new_with_symbolic(self.sparsity_pattern.clone().unwrap(), a.as_ref())
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
    fn solve(&mut self, a: &SparseColMat<usize, dtype>, b: &Mat<dtype>) -> Mat<dtype> {
        let ata = a
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(a.as_ref());
        let atb = a.as_ref().transpose().mul(b);

        if self.sparsity_pattern.is_none() {
            self.sparsity_pattern = Some(solvers::SymbolicLu::try_new(ata.symbolic()).unwrap());
        }

        solvers::Lu::try_new_with_symbolic(self.sparsity_pattern.clone().unwrap(), ata.as_ref())
            .unwrap()
            .solve(&atb)
    }
}

#[cfg(test)]
mod test {
    use faer::mat;
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
        let x = solver.solve(&a, &b);
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
