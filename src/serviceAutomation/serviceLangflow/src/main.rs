use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "langflow",
        "Langflow",
        "automation",
        "workflow-ui",
        "vendor/layerAutomation/langflow",
        "src/serviceAutomation/serviceLangflow",
        8301,
    )
    .with_upstream("http://langflow:7860", Some("/health"))
    .with_tags(&["workflow", "builder"])
    .apply_env();

    run(config).await
}
