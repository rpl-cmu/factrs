use samrs::containers::{Symbol, Values, X};
use samrs::factors::Factor;
use samrs::linalg::dvector;
use samrs::make_bundle;
use samrs::noise::GaussianNoise;
use samrs::residuals::PriorResidual;
use samrs::robust::{GemanMcClure, Tukey, L2};
use samrs::variables::{Variable, VariableEnum, Vector3, SO3};

make_bundle!(
    NewBundle;
    Symbol;
    VariableEnum;
    REnum: L2, GemanMcClure, Tukey;
    GaussianNoise;
    PriorResidual<SO3>;
);

#[allow(dead_code)]
fn main() {
    let xi = dvector![0.1, 0.2, 0.3];

    let r = SO3::exp(xi.as_view());
    let v = Vector3::new(1.0, 2.0, 3.0);
    let e: VariableEnum = r.clone().into();

    let mut values: Values<Symbol, VariableEnum> = Values::new();
    values.insert(X(0), r.clone());
    values.insert(X(1), v);
    values.insert(X(2), e);
    println!("{:#?}", values);

    // let temp = values.get_mut(&X(0)).unwrap();
    // *temp = temp.oplus(&xi);

    // let so3s: Vec<&SO3> = values.filter().collect();
    // println!("{:#?}", so3s);

    // let vecs: Vec<Vector3> = values.into_filter().collect();
    // println!("{:#?}", vecs);

    // : Factor<Symbol, SO3, PriorResidual<SO3>, GaussianNoise, L2>

    let f = Factor::<NewBundle>::new(vec![X(0)], PriorResidual::new(&r))
        .set_noise(GaussianNoise::from_scalar_sigma(1e-2, r.dim()))
        .set_robust(Tukey::default())
        .build();

    f.error(&values);
}
