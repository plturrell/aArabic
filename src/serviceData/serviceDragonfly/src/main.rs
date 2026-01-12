use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "dragonfly",
        "DragonflyDB",
        "data",
        "cache",
        "vendor/layerData/dragonflydb",
        "src/serviceData/serviceDragonfly",
        8203,
    )
    .with_upstream("redis://dragonfly:6379", Some("redis://dragonfly:6379"))
    .with_tags(&["cache", "redis"])
    .apply_env();

    run(config).await
}
