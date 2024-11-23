use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_quote, ItemImpl};

pub fn tag(mut item: ItemImpl) -> TokenStream2 {
    // Parse what residual number we're using
    let residual_trait = item.trait_.clone().expect("No trait was given").1;
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

    // If we shouldd add typetag
    let typetag = if cfg!(feature = "serde") {
        // Add where clauses to all impl
        let all_type_params: Vec<_> = item.generics.type_params().cloned().collect();
        for type_param in all_type_params {
            let ident = &type_param.ident;
            item.generics.make_where_clause();
            item.generics
                .where_clause
                .as_mut()
                .unwrap()
                .predicates
                .push(parse_quote!(#ident: typetag::Tagged));
        }

        quote!( #[typetag::serde] )
    } else {
        TokenStream2::new()
    };

    // Build the output
    let generics = &item.generics;
    let self_ty = &item.self_ty;
    let where_clause = &generics.where_clause;

    quote! {
        #item

        #typetag
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
