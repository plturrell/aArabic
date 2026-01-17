use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "keycloak",
        "Keycloak",
        "core",
        "identity",
        "vendor/layerCore/keycloak",
        "src/serviceCore/serviceKeycloak",
        8102,
    )
    .with_upstream("http://keycloak:8080", Some("/health/ready"))
    .with_tags(&["auth", "oidc"])
    .with_dependencies(&["keycloak-db"])
    .apply_env();

    run(config).await
}
