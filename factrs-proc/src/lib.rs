use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse::Parser, parse_macro_input, punctuated::Punctuated, Item};

mod fac;

#[proc_macro]
pub fn fac(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let factor = parse_macro_input!(input as fac::Factor);

    fac::fac(factor).into()
}

#[proc_macro_attribute]
pub fn mark_residual(
    args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    mark_residual2(args.into(), input.into()).into()
}

fn mark_residual2(args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse arguments
    let args = Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated
        .parse2(args)
        .unwrap();

    let prefix = if args.len() == 1 && args[0] == "internal" {
        quote! { crate }
    } else {
        quote! { factrs }
    };

    // Parse the item
    let item = syn::parse2(input.clone()).expect("Failed to parse residual implementation.");
    let impl_block = match item {
        Item::Impl(s) => s,
        _ => panic!("Not an impl"),
    };

    // Parse what residual number we're using
    let residual_trait = impl_block.trait_.clone().expect("No trait was given").1;
    let num = residual_trait
        .segments
        .last()
        .expect("No ident found")
        .ident
        .to_string()
        .replace("Residual", "")
        .parse::<u32>()
        .expect("Residual wasn't parseable as a number");

    // Build all the things we need from it
    let residual_values = format!("residual{}_values", num);
    let residual_values = syn::Ident::new(&residual_values, proc_macro2::Span::call_site());
    let residual_jacobian = format!("residual{}_jacobian", num);
    let residual_jacobian = syn::Ident::new(&residual_jacobian, proc_macro2::Span::call_site());

    // Build the output
    let generics = impl_block.generics;
    let self_ty = impl_block.self_ty;
    let where_clause = &generics.where_clause;

    quote! {
        #input

        impl #generics #prefix::residuals::Residual for #self_ty #where_clause {
            fn dim_in(&self) -> usize {
                <<Self as #residual_trait>::DimIn as #prefix::linalg::DimName>::USIZE
            }

            fn dim_out(&self) -> usize {
                <<Self as  #residual_trait>::DimOut as #prefix::linalg::DimName>::USIZE
            }

            fn residual(&self, values: &#prefix::containers::Values, keys: &[#prefix::containers::Key]) -> #prefix::linalg::VectorX {
                #residual_trait::#residual_values(self, values, keys)
            }

            fn residual_jacobian(&self, values: &#prefix::containers::Values, keys: &[#prefix::containers::Key]) -> #prefix::linalg::DiffResult<#prefix::linalg::VectorX, #prefix::linalg::MatrixX> {
                #residual_trait::#residual_jacobian(self, values, keys)
            }
        }
    }
}
