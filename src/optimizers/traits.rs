use crate::{
    containers::{Graph, Values},
    dtype,
};

#[derive(Debug)]
pub enum OptError {
    MaxIterations(Values),
    InvalidSystem,
    FailedToStep,
}

pub type OptResult = Result<Values, OptError>;

pub struct OptimizerParams {
    pub max_iterations: usize,
    pub error_tol_relative: dtype,
    pub error_tol_absolute: dtype,
    pub error_tol: dtype,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            error_tol_relative: 1e-6,
            error_tol_absolute: 1e-6,
            error_tol: 0.0,
        }
    }
}

pub trait Optimizer {
    fn new(graph: Graph) -> Self;

    fn graph(&self) -> &Graph;

    fn params(&self) -> &OptimizerParams;

    fn step(&mut self, values: Values) -> OptResult;

    fn error(&self, values: &Values) -> dtype {
        self.graph().error(values)
    }

    // TODO: Custom logging based on optimizer
    fn optimize(&mut self, mut values: Values) -> OptResult {
        // Check if we need to optimize at all
        let mut error_old = self.error(&values);
        if error_old <= self.params().error_tol {
            log::info!("Error is already below tolerance, skipping optimization");
            return Ok(values);
        }

        log::info!(
            "{:^5} | {:^12} | {:^12} | {:^12}",
            "Iter",
            "Error",
            "ErrorAbs",
            "ErrorRel"
        );
        log::info!(
            "{:^5} | {:^12} | {:^12} | {:^12}",
            "-----",
            "------------",
            "------------",
            "------------"
        );
        log::info!(
            "{:^5} | {:^12.4e} | {:^12} | {:^12}",
            0,
            error_old,
            "-",
            "-"
        );

        // Begin iterations
        let mut error_new = error_old;
        for i in 1..self.params().max_iterations + 1 {
            error_old = error_new;
            values = self.step(values)?;

            // Evaluate error again to see how we did
            error_new = self.error(&values);

            let error_decrease_abs = error_old - error_new;
            let error_decrease_rel = error_decrease_abs / error_old;

            log::info!(
                "{:^5} | {:^12.4e} | {:^12.4e} | {:^12.4e}",
                i,
                error_new,
                error_decrease_abs,
                error_decrease_rel
            );

            // Check if we need to stop
            if error_new <= self.params().error_tol {
                log::info!("Error is below tolerance, stopping optimization");
                return Ok(values);
            }
            if error_decrease_abs <= self.params().error_tol_absolute {
                log::info!("Error decrease is below absolute tolerance, stopping optimization");
                return Ok(values);
            }
            if error_decrease_rel <= self.params().error_tol_relative {
                log::info!("Error decrease is below relative tolerance, stopping optimization");
                return Ok(values);
            }
        }

        Err(OptError::MaxIterations(values))
    }
}
