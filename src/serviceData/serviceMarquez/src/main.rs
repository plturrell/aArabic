use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "marquez",
        "Marquez",
        "data",
        "lineage",
        "vendor/layerData/marquez",
        "src/serviceData/serviceMarquez",
        8204,
    )
    .with_upstream("http://marquez:5000", Some("/api/v1/namespaces"))
    .with_tags(&["lineage", "metadata"])
    .with_dependencies(&["marquez-db"])
    .apply_env();

    run(config).await
}
