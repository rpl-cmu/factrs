use proc_macro2::Span;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_quote;
use syn::{parse::Parse, punctuated::Punctuated, Expr, Ident, Token};

pub struct Factor {
    residual: Expr,
    keys: Punctuated<Expr, Token![,]>,
    noise: Option<Expr>,
    robust: Option<Expr>,
}

impl Factor {
    fn noise_call(&self) -> TokenStream2 {
        match &self.noise {
            Some(n) => quote! { .noise(#n) },
            None => TokenStream2::new(),
        }
    }

    fn robust_call(&self) -> TokenStream2 {
        match &self.robust {
            Some(r) => quote! {.robust(#r) },
            None => TokenStream2::new(),
        }
    }

    fn new_call(&self) -> TokenStream2 {
        let func = Ident::new(&format!("new{}", self.keys.len()), Span::call_site());
        let res = &self.residual;
        let keys = &self.keys;
        quote! { #func(#res, #keys) }
    }
}

impl Parse for Factor {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let input = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;

        // Make sure we go the right number of arguments
        if input.len() < 2 {
            return Err(syn::Error::new_spanned(
                &input[0],
                "Expected at least two items",
            ));
        } else if input.len() > 4 {
            return Err(syn::Error::new_spanned(
                input.last(),
                "Expected at most four items",
            ));
        }

        // Residual is first
        let residual = input[0].clone();

        // Then the keys
        let keys = match &input[1] {
            // in brackets
            Expr::Array(a) => a.elems.clone(),
            // in parentheses
            Expr::Tuple(t) => t.elems.clone(),
            // a single key for unary factors
            Expr::Path(_) | Expr::Call(_) => {
                let mut p = Punctuated::<Expr, Token![,]>::new();
                p.push(input[1].clone());
                p
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    &input[1],
                    "Expected keys in brackets or parentheses",
                ));
            }
        };

        // Then the noise
        let noise = if input.len() >= 3 {
            match &input[2] {
                Expr::Lit(l) => {
                    Some(parse_quote!(factrs::noise::GaussianNoise::from_scalar_sigma(#l)))
                }
                _ => Some(input[2].clone()),
            }
        } else {
            None
        };

        // Finally robust cost function
        let robust = if input.len() == 4 {
            Some(input[3].clone())
        } else {
            None
        };

        Ok(Factor {
            residual,
            keys,
            noise,
            robust,
        })
    }
}

pub fn fac(factor: Factor) -> TokenStream2 {
    let call = factor.new_call();
    let noise = factor.robust_call();
    let robust = factor.noise_call();

    let out = quote! {
        factrs::containers::FactorBuilder:: #call #noise #robust.build()
    };

    out
}
