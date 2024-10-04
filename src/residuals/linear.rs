use std::fmt;

use super::Residual;
use crate::{
    containers::{Factor, Key, Values},
    linalg::{DiffResult, MatrixX, VectorX},
    linear::LinearFactor,
    noise::UnitNoise,
    robust::L2,
    tag_residual,
};

tag_residual!(LinearResidual);

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearResidual {
    factor: LinearFactor,
    lin_point: Values,
}

impl fmt::Display for LinearResidual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LinearResidual(factor: {:?}, lin_point: {})",
            self.factor, self.lin_point
        )
    }
}

impl LinearResidual {
    pub fn new(factor: LinearFactor, lin_point: Values) -> Self {
        Self { factor, lin_point }
    }

    // TODO: Test this, make sure we're constructing it correctly
    // TODO: Add in asserts to check?
    pub fn into_factor(self) -> Factor {
        Factor::new_unchecked(
            self.factor.keys.clone(),
            Box::new(self),
            // UnitNoise doesn't actually have any code that is size dependent
            Box::new(UnitNoise::<1>),
            Box::new(L2),
        )
    }
}

impl Residual for LinearResidual {
    fn dim_in(&self) -> usize {
        self.factor.dim_in()
    }

    fn dim_out(&self) -> usize {
        self.factor.dim_out()
    }

    fn residual(&self, values: &Values, _keys: &[Key]) -> VectorX {
        // Keys should always match up, as we construct the factor from the keys
        let delta = self.lin_point.ominus(values);
        self.factor.residual(&delta)
    }

    fn residual_jacobian(&self, values: &Values, keys: &[Key]) -> DiffResult<VectorX, MatrixX> {
        let value = self.residual(values, keys);
        let diff = self.factor.jacobian().into_owned();
        DiffResult { value, diff }
    }
}
