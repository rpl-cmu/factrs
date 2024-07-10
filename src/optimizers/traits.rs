use crate::{containers::Graph, dtype};

#[derive(Debug)]
pub enum OptError {
    MaxIterations,
    InvalidSystem,
    FailedToStep,
}

pub type OptResult<Input> = Result<Input, OptError>;

// ------------------------- Optimizer Params ------------------------- //
pub struct OptParams {
    pub max_iterations: usize,
    pub error_tol_relative: dtype,
    pub error_tol_absolute: dtype,
    pub error_tol: dtype,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            error_tol_relative: 1e-6,
            error_tol_absolute: 1e-6,
            error_tol: 0.0,
        }
    }
}

// ------------------------- Optimizer Observers ------------------------- //
pub trait OptObserver {
    type Input;
    fn on_step(&self, values: &Self::Input, time: f64);
}

pub struct OptObserverVec<I> {
    observers: Vec<Box<dyn OptObserver<Input = I>>>,
}

impl<I> OptObserverVec<I> {
    pub fn add(&mut self, callback: impl OptObserver<Input = I> + 'static) {
        let boxed = Box::new(callback);
        self.observers.push(boxed);
    }

    pub fn notify(&self, values: &I, idx: usize) {
        for callback in &self.observers {
            callback.on_step(values, idx as f64);
        }
    }
}

impl<I> Default for OptObserverVec<I> {
    fn default() -> Self {
        Self {
            observers: Vec::new(),
        }
    }
}

// ------------------------- Actual Trait Impl ------------------------- //
pub trait Optimizer {
    type Input;

    // Wrappers for setup
    fn observers(&self) -> &OptObserverVec<Self::Input>;

    fn params(&self) -> &OptParams;

    // Core optimization functions
    fn step(&mut self, values: Self::Input) -> OptResult<Self::Input>;

    fn error(&self, values: &Self::Input) -> dtype;

    fn init(&mut self, _values: &Self::Input) {}

    // TODO: Custom logging based on optimizer
    fn optimize(&mut self, mut values: Self::Input) -> OptResult<Self::Input> {
        // Setup up everything from our values
        self.init(&values);

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

            // Notify observers
            self.observers().notify(&values, i);

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

        Err(OptError::MaxIterations)
    }
}

pub trait GraphOptimizer: Optimizer {
    fn new(graph: Graph) -> Self;

    fn graph(&self) -> &Graph;
}
