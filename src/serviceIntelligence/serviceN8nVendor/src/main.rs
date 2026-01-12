use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "n8n",
        "n8n",
        "intelligence",
        "automation",
        "vendor/layerIntelligence/n8n",
        "src/serviceIntelligence/serviceN8nVendor",
        8403,
    )
    .with_upstream("http://n8n:5678", Some("/healthz"))
    .with_tags(&["automation", "workflow"])
    .apply_env();

    run(config).await
}
