use crate::{dtype, make_enum_robust};

// TODO: Consider changing names to \rho and w
pub trait RobustCost: Sized {
    fn loss(&self, d2: dtype) -> dtype;

    fn weight(&self, d2: dtype) -> dtype;
}

// ------------------------- L2 Norm ------------------------- //
pub struct L2;

impl Default for L2 {
    fn default() -> Self {
        L2
    }
}

impl RobustCost for L2 {
    fn loss(&self, d2: dtype) -> dtype {
        d2 / 2.0
    }

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
    fn loss(&self, d2: dtype) -> dtype {
        d2.sqrt()
    }

    fn weight(&self, d2: dtype) -> dtype {
        if d2 <= 1e-3 {
            1.0
        } else {
            1.0 / d2.sqrt()
        }
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
    fn loss(&self, d2: dtype) -> dtype {
        if d2 <= self.k * self.k {
            d2 / 2.0
        } else {
            let d = d2.sqrt();
            self.k * (d - self.k / 2.0)
        }
    }

    fn weight(&self, d2: dtype) -> dtype {
        let dabs = d2.sqrt();
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
    fn loss(&self, d2: dtype) -> dtype {
        let d = d2.sqrt();
        self.c * self.c * (d / self.c - (1.0 + (d / self.c)).ln())
    }

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
    fn loss(&self, d2: dtype) -> dtype {
        self.c2 * ((1.0 + d2 / self.c2).ln()) / 2.0
    }

    fn weight(&self, d2: dtype) -> dtype {
        1.0 / (1.0 + d2 / self.c2)
    }
}

// ------------------------- Geman-McClure ------------------------- //
pub struct GemanMcClure {
    c2: dtype,
}

impl GemanMcClure {
    pub fn new(c: dtype) -> Self {
        GemanMcClure { c2: c * c }
    }
}

impl Default for GemanMcClure {
    fn default() -> Self {
        GemanMcClure {
            c2: 1.3998 * 1.3998,
        }
    }
}

impl RobustCost for GemanMcClure {
    fn loss(&self, d2: dtype) -> dtype {
        0.5 * self.c2 * d2 / (self.c2 + d2)
    }

    fn weight(&self, d2: dtype) -> dtype {
        let denom = self.c2 + d2;
        let frac = self.c2 / denom;
        frac * frac
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
    fn loss(&self, d2: dtype) -> dtype {
        self.c2 * (1.0 - (-d2 / self.c2).exp()) / 2.0
    }

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
    fn loss(&self, d2: dtype) -> dtype {
        if d2 <= self.c2 {
            self.c2 * (1.0 - (1.0 - d2 / self.c2).powi(3)) / 6.0
        } else {
            self.c2 / 6.0
        }
    }

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

#[cfg(test)]
mod test {
    use matrixcompare::assert_scalar_eq;

    use super::*;
    use crate::linalg::num_derivative;

    fn test_weight(robust: &impl RobustCost, d: dtype) {
        let got = robust.weight(d * d);
        // weight = loss'(d) / d
        let actual = num_derivative(|d| robust.loss(d * d), d) / d;

        println!("Weight got: {}, Weight actual: {}", got, actual);
        assert_scalar_eq!(got, actual, comp = abs, tol = 1e-6);
    }

    macro_rules! robust_tests {
        ($($robust:ident),*) => {
            use paste::paste;

            paste!{
                $(
                    #[test]
                    #[allow(non_snake_case)]
                    fn [<$robust _weight>]() {
                        let robust = $robust::default();
                        // Test near origin
                        test_weight(&robust, 0.1);
                        // Test far away
                        test_weight(&robust, 50.0);
                    }

                    #[test]
                    #[allow(non_snake_case)]
                    fn [<$robust _center>]() {
                        let robust = $robust::default();
                        println!("Center: {}", robust.loss(0.0));
                        assert_scalar_eq!(robust.loss(0.0), 0.0, comp=float);
                    }

                )*
            }

        }
    }

    robust_tests!(L2, L1, Huber, Fair, Cauchy, GemanMcClure, Welsch, Tukey);
}
