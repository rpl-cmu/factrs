use proc_macro2::Ident;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::ConstParam;
use syn::Error;
use syn::GenericParam;
use syn::{ItemImpl, Type, TypePath};

fn type_name(mut ty: &Type) -> Option<Ident> {
    loop {
        match ty {
            Type::Path(TypePath { qself: None, path }) => {
                return Some(path.segments.last().unwrap().ident.clone());
            }
            Type::Group(group) => {
                ty = &group.elem;
            }
            _ => return None,
        }
    }
}

pub fn tag(item: ItemImpl) -> TokenStream2 {
    if !cfg!(feature = "serde") {
        return quote! { #item };
    }

    let mut expanded = quote::quote! {
        #item
    };

    // Find name
    let name = type_name(&item.self_ty).unwrap().to_token_stream();
    let name_str = name.to_string();
    let name_quotes = quote!(#name_str);

    // Have to do three things here to get typetag to work
    // - Impl typetag::Tagged
    // - Register our variable in typetag
    // - Tag PriorResidual, BetweenResidual with this type
    match item.generics.params.len() {
        0 => {
            let msg = "variable should have dtype generic";
            return Error::new_spanned(&item.generics, msg).to_compile_error();
        }
        // Simple variable, just use as is
        1 => {
            expanded.extend(tag_all(&name));
            expanded.extend(quote!(
                impl typetag::Tagged for #name {
                    fn tag() -> String {
                        String::from(#name_quotes)
                    }
                }
            ));
        }
        // With a single const generic
        2 => {
            let first_generic = item.generics.params.first().unwrap();
            if let GenericParam::Const(ConstParam { ident, .. }) = first_generic {
                let format = quote! { #name<{}> }.to_string();
                expanded.extend(quote! {
                    impl<const #ident: usize> typetag::Tagged for #name<#ident> {
                        fn tag() -> String {
                            format!(#format, #ident)
                        }
                    }
                });
                for i in 1usize..=20 {
                    let name_num = quote!(#name<#i>);
                    expanded.extend(tag_all(&name_num));
                }
            }
        }
        // Anymore and it's up to the user
        _ => {}
    }

    expanded
}

fn tag_all(kind: &TokenStream2) -> TokenStream2 {
    quote! {
        // Self
        typetag::__private::inventory::submit! {
            <dyn factrs::variables::VariableSafe>::typetag_register(
                stringify!(#kind),
                (|deserializer| typetag::__private::Result::Ok(
                    typetag::__private::Box::new(
                        typetag::__private::erased_serde::deserialize::<#kind>(deserializer)?
                    ),
                )) as typetag::__private::DeserializeFn<<dyn factrs::variables::VariableSafe as typetag::__private::Strictest>::Object>,
            )
        }

        // Prior
        typetag::__private::inventory::submit! {
            <dyn factrs::residuals::Residual>::typetag_register(
                stringify!(PriorResidual<#kind>),
                (|deserializer| typetag::__private::Result::Ok(
                    typetag::__private::Box::new(
                        typetag::__private::erased_serde::deserialize::<factrs::residuals::PriorResidual<#kind>>(deserializer)?
                    ),
                )) as typetag::__private::DeserializeFn<<dyn factrs::residuals::Residual as typetag::__private::Strictest>::Object>,
            )
        }

        // Between
        typetag::__private::inventory::submit! {
            <dyn factrs::residuals::Residual>::typetag_register(
                stringify!(PriorResidual<#kind>),
                (|deserializer| typetag::__private::Result::Ok(
                    typetag::__private::Box::new(
                        typetag::__private::erased_serde::deserialize::<factrs::residuals::PriorResidual<#kind>>(deserializer)?
                    ),
                )) as typetag::__private::DeserializeFn<<dyn factrs::residuals::Residual as typetag::__private::Strictest>::Object>,
            )
        }
    }
}
