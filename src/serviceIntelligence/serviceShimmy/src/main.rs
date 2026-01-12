use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "shimmy",
        "Shimmy",
        "intelligence",
        "orchestration",
        "vendor/layerIntelligence/shimmy-ai",
        "src/serviceIntelligence/serviceShimmy",
        8401,
    )
    .with_upstream("http://shimmy:3001", Some("/health"))
    .with_tags(&["workflow", "llm"])
    .apply_env();

    run(config).await
}
