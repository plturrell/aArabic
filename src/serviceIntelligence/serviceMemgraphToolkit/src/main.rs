use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "memgraph-ai-toolkit",
        "Memgraph AI Toolkit",
        "intelligence",
        "library",
        "vendor/layerIntelligence/memgraph-ai-toolkit",
        "src/serviceIntelligence/serviceMemgraphToolkit",
        8404,
    )
    .with_tags(&["graph", "ai"])
    .apply_env();

    run(config).await
}
