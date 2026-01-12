use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "qdrant",
        "Qdrant",
        "data",
        "vector-db",
        "vendor/layerData/qdrant",
        "src/serviceData/serviceQdrant",
        8202,
    )
    .with_upstream("http://qdrant:6333", Some("/readyz"))
    .with_tags(&["vector", "search"])
    .apply_env();

    run(config).await
}
