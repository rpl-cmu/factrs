use crate::{dtype, linalg::VectorX, make_enum_robust};

pub trait RobustCost: Sized {
    fn weight(&self, d2: dtype) -> dtype;

    fn weight_vec(&self, r: &VectorX) -> VectorX {
        r * self.weight(r.norm_squared()).sqrt()
    }
}

// ------------------------- L2 Norm ------------------------- //
pub struct L2;

impl Default for L2 {
    fn default() -> Self {
        L2
    }
}

impl RobustCost for L2 {
    fn weight(&self, _d: dtype) -> dtype {
        1.0
    }
}

// ------------------------- L1 Norm ------------------------- //
pub struct L1;

impl Default for L1 {
    fn default() -> Self {
        L1
    }
}

impl RobustCost for L1 {
    fn weight(&self, d2: dtype) -> dtype {
        1.0 / d2.sqrt().abs()
    }
}

// ------------------------- Huber ------------------------- //
pub struct Huber {
    k: dtype,
}

impl Huber {
    pub fn new(k: dtype) -> Self {
        Huber { k }
    }
}

impl Default for Huber {
    fn default() -> Self {
        Huber { k: 1.345 }
    }
}

impl RobustCost for Huber {
    fn weight(&self, d2: dtype) -> dtype {
        let dabs = d2.sqrt().abs();
        if dabs <= self.k {
            1.0
        } else {
            self.k / dabs
        }
    }
}

// ------------------------- Fair ------------------------- //
pub struct Fair {
    c: dtype,
}

impl Fair {
    pub fn new(c: dtype) -> Self {
        Fair { c }
    }
}

impl Default for Fair {
    fn default() -> Self {
        Fair { c: 1.3998 }
    }
}

impl RobustCost for Fair {
    fn weight(&self, d: dtype) -> dtype {
        1.0 / (1.0 + d.sqrt().abs() / self.c)
    }
}

// ------------------------- Cauchy ------------------------- //
pub struct Cauchy {
    c2: dtype,
}

impl Cauchy {
    pub fn new(c: dtype) -> Self {
        Cauchy { c2: c * c }
    }
}

impl Default for Cauchy {
    fn default() -> Self {
        Cauchy {
            c2: 2.3849 * 2.3849,
        }
    }
}

impl RobustCost for Cauchy {
    fn weight(&self, d2: dtype) -> dtype {
        1.0 / (1.0 + d2 / self.c2)
    }
}

// ------------------------- Geman-McClure ------------------------- //
// TODO: Generalized Geman-McClure?
pub struct GemanMcClure {
    c: dtype,
}

impl GemanMcClure {
    pub fn new(c: dtype) -> Self {
        GemanMcClure { c }
    }
}

impl Default for GemanMcClure {
    fn default() -> Self {
        GemanMcClure { c: 1.3998 }
    }
}

impl RobustCost for GemanMcClure {
    fn weight(&self, d: dtype) -> dtype {
        1.0 / (1.0 + (d / self.c).powi(2)).sqrt()
    }
}

// ------------------------- Welsch ------------------------- //
pub struct Welsch {
    c2: dtype,
}

impl Welsch {
    pub fn new(c: dtype) -> Self {
        Welsch { c2: c * c }
    }
}

impl Default for Welsch {
    fn default() -> Self {
        Welsch {
            c2: 2.9846 * 2.9846,
        }
    }
}

impl RobustCost for Welsch {
    fn weight(&self, d2: dtype) -> dtype {
        (-d2 / self.c2).exp()
    }
}

// ------------------------- Tukey ------------------------- //
pub struct Tukey {
    c2: dtype,
}

impl Tukey {
    pub fn new(c: dtype) -> Self {
        Tukey { c2: c * c }
    }
}

impl Default for Tukey {
    fn default() -> Self {
        Tukey {
            c2: 4.6851 * 4.6851,
        }
    }
}

impl RobustCost for Tukey {
    fn weight(&self, d2: dtype) -> dtype {
        if d2 <= self.c2 {
            (1.0 - d2 / self.c2).powi(2)
        } else {
            0.0
        }
    }
}

// ------------------------- Make Enum ------------------------- //

mod macros;

make_enum_robust!(
    RobustEnum,
    L2,
    L1,
    Huber,
    Fair,
    Cauchy,
    GemanMcClure,
    Welsch,
    Tukey
);
