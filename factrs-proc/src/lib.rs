use proc_macro2::TokenStream;
use quote::quote;
use syn::Item;

#[proc_macro_attribute]
pub fn mark_residual(
    args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    mark_residual2(args.into(), input.into()).into()
}

fn mark_residual2(_args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse the input
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

        impl #generics factrs::residuals::Residual for #self_ty #where_clause {
            fn dim_in(&self) -> usize {
                <<Self as #residual_trait>::DimIn as factrs::linalg::DimName>::USIZE
            }

            fn dim_out(&self) -> usize {
                <<Self as  #residual_trait>::DimOut as factrs::linalg::DimName>::USIZE
            }

            fn residual(&self, values: &factrs::containers::Values, keys: &[factrs::containers::Key]) -> factrs::linalg::VectorX {
                #residual_trait::#residual_values(self, values, keys)
            }

            fn residual_jacobian(&self, values: &factrs::containers::Values, keys: &[factrs::containers::Key]) -> factrs::linalg::DiffResult<factrs::linalg::VectorX, factrs::linalg::MatrixX> {
                #residual_trait::#residual_jacobian(self, values, keys)
            }
        }
    }
}
