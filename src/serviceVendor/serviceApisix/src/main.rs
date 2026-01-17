use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "apisix",
        "APISIX Gateway",
        "core",
        "gateway",
        "vendor/layerCore/apisix",
        "src/serviceCore/serviceApisix",
        8101,
    )
    .with_upstream("http://gateway:9080", Some("/apisix/status"))
    .with_tags(&["gateway", "oidc"])
    .with_dependencies(&["keycloak"])
    .apply_env();

    run(config).await
}
