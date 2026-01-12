use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "memgraph",
        "Memgraph",
        "data",
        "graph-db",
        "vendor/layerData/memgraph",
        "src/serviceData/serviceMemgraph",
        8201,
    )
    .with_upstream("bolt://memgraph:7687", Some("bolt://memgraph:7687"))
    .with_tags(&["graph", "bolt"])
    .apply_env();

    run(config).await
}
