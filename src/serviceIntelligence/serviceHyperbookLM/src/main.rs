use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "hyperbooklm",
        "HyperbookLM",
        "intelligence",
        "research-ui",
        "vendor/layerIntelligence/hyperbooklm",
        "src/serviceIntelligence/serviceHyperbookLM",
        8402,
    )
    .with_upstream("http://hyperbooklm:3002", Some("/"))
    .with_tags(&["research", "ui"])
    .apply_env();

    run(config).await
}
