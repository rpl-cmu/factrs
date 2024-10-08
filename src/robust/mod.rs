//! Robust kernels (M-Estimators) for robust estimation.
//!
//! We recommend checking out "Parameter Estimation Techniques:  A Tutorial with
//! Application to Conic Fitting" by Zhengyou Zhang for details on our
//! implementations.
//!
//! They can roughly be split into the following categories:
//!
//! | Name         | Loss Function | Weight Function | Asymptotic Behavior |
//! |--------------|---------------|-----------------|---------------------|
//! | L2           | $x^2 / 2$ | $1$ | Quadratic           |
//! | L1           | $\|x\|$ | $1 / \|x\|$ | Linear              |
//! | Huber $\begin{cases} \|x\| \leq k \\\\ \|x\| > k \end{cases}$ | $\begin{cases} x^2/2 \\\\ k(\|x\| - k/2) \end{cases}$ | $\begin{cases} 1 \\\\ k/\|x\| \end{cases}$ | Linear              |
//! | Fair         | $c^2 \left(\frac{\|x\|}{c} - \ln(1 + \frac{\|x\|}{c})\right)$ | $1 / (1 + \frac{\|x\|}{c})$ | Linear              |
//! | Cauchy       | $\frac{c^2}{2}\ln\left(1 + (x/c)^2\right)$ | $1 / (1 + (x/c)^2)$ | Constant            |
//! | Geman-McClure| $\frac{c^2 x^2}{2} / (c^2 + x^2)$ | $c^2 / (c^2 + x^2)^2$ | Constant            |
//! | Welsch       | $\frac{c^2}{2}\left(1 - \exp(-(x/c)^2)\right)$ | $\exp(-(x/c)^2)$ | Constant            |
//! | Tukey $\begin{cases} \|x\| \leq c \\\\ \|x\| > c \end{cases}$ | $\begin{cases} \frac{c^2}{6}\left(1 - \left(1 - (x/c)^2\right)^3\right) \\\\ \frac{c^2}{6} \end{cases}$ | $\begin{cases} \left(1 - (x/c)^2\right)^2 \\\\ 0 \end{cases}$ | Constant            |
//!
//! Generally constant asymptotic behavior is the best at outlier rejection, but
//! relies heavily on good initialization. Some work, such as Graduated
//! Non-Convexity (GNC), has been shown to circumvent this requirement.

use std::fmt::Debug;

use crate::dtype;

/// Robust cost function
///
/// Represents a robust cost function \rho. Note that most robust cost functions
/// use x^2 in some form, so rather than passing x, we pass x^2. If you'd like
/// to implement your own kernel, we recommend using
/// [NumericalDiff](crate::linalg::NumericalDiff) to check that the weight is
/// correct.

pub trait RobustCost: Default + Debug {
    /// Compute the loss \rho(x^2)
    fn loss(&self, d2: dtype) -> dtype;

    /// Compute the weight \rho'(x^2) / x
    fn weight(&self, d2: dtype) -> dtype;
}

/// Safe version of [RobustCost] that can be used in trait objects
///
/// This will be implemented via a blanket impl for all types that implement
/// [RobustCost]. It's actually redundant, but we include it for completeness in
/// case we want to add additional methods in the future.
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait RobustCostSafe: Debug {
    fn loss(&self, d2: dtype) -> dtype;

    fn weight(&self, d2: dtype) -> dtype;
}

impl<
        #[cfg(not(feature = "serde"))] T: RobustCost,
        #[cfg(feature = "serde")] T: RobustCost + crate::serde::Tagged,
    > RobustCostSafe for T
{
    fn loss(&self, d2: dtype) -> dtype {
        RobustCost::loss(self, d2)
    }

    fn weight(&self, d2: dtype) -> dtype {
        RobustCost::weight(self, d2)
    }

    #[doc(hidden)]
    #[cfg(feature = "serde")]
    fn typetag_name(&self) -> &'static str {
        Self::TAG
    }

    #[doc(hidden)]
    #[cfg(feature = "serde")]
    fn typetag_deserialize(&self) {}
}

/// Register a type as a [robust cost](crate::robust) for serialization.
#[macro_export]
macro_rules! tag_robust {
    ($($ty:ty),* $(,)?) => {$(
        $crate::register_typetag!($crate::robust::RobustCostSafe, $ty);
    )*};
}

tag_robust!(L2, L1, Huber, Fair, Cauchy, GemanMcClure, Welsch, Tukey);

// ------------------------- L2 Norm ------------------------- //
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg(test)]
mod test {
    use matrixcompare::assert_scalar_eq;

    use super::*;
    use crate::linalg::numerical_derivative;

    #[cfg(not(feature = "f32"))]
    const EPS: dtype = 1e-6;
    #[cfg(not(feature = "f32"))]
    const TOL: dtype = 1e-6;

    #[cfg(feature = "f32")]
    const EPS: dtype = 1e-3;
    #[cfg(feature = "f32")]
    const TOL: dtype = 1e-2;

    fn test_weight(robust: &impl RobustCost, d: dtype) {
        let got = robust.weight(d * d);
        // weight = loss'(d) / d
        let actual = numerical_derivative(|d| robust.loss(d * d), d, EPS).diff / d;

        println!("Weight got: {}, Weight actual: {}", got, actual);
        assert_scalar_eq!(got, actual, comp = abs, tol = TOL);
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
                        println!("Center: {}", RobustCost::loss(&robust, 0.0));
                        assert_scalar_eq!(RobustCost::loss(&robust, 0.0), 0.0, comp=float);
                    }

                )*
            }

        }
    }

    robust_tests!(L2, L1, Huber, Fair, Cauchy, GemanMcClure, Welsch, Tukey);
}
