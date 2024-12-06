use proc_macro2::Ident;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::parse_quote;
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

pub fn mark(item: ItemImpl) -> TokenStream2 {
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
            let name_str = name.to_string();
            expanded.extend(tag_all(&name, &name_str));
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
                let format = format!("{}<{{}}>", name.to_string());
                // let format = quote! { #name<{}> }.to_string();
                expanded.extend(quote! {
                    impl<const #ident: usize> typetag::Tagged for #name<#ident> {
                        fn tag() -> String {
                            format!(#format, #ident)
                        }
                    }
                });
                for i in 1usize..=20 {
                    let name_str = format!("{}<{}>", name, i);
                    let name_qt = parse_quote!(#name<#i>);
                    expanded.extend(tag_all(&name_qt, &name_str));
                }
            }
        }
        // Anymore and it's up to the user
        // TODO: Could at least implement tagged here
        _ => {}
    }

    expanded
}

fn tag_all(kind: &TokenStream2, name: &str) -> TokenStream2 {
    let name_prior = format!("PriorResidual<{}>", name);
    let name_between = format!("BetweenResidual<{}>", name);
    quote! {
        // Self
        typetag::__private::inventory::submit! {
            <dyn factrs::variables::VariableSafe>::typetag_register(
                #name,
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
                #name_prior,
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
                #name_between,
                (|deserializer| typetag::__private::Result::Ok(
                    typetag::__private::Box::new(
                        typetag::__private::erased_serde::deserialize::<factrs::residuals::PriorResidual<#kind>>(deserializer)?
                    ),
                )) as typetag::__private::DeserializeFn<<dyn factrs::residuals::Residual as typetag::__private::Strictest>::Object>,
            )
        }
    }
}
