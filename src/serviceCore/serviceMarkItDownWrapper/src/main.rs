use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "markitdown",
        "MarkItDown",
        "core",
        "document-converter",
        "vendor/layerCore/markitdown",
        "src/serviceCore/serviceMarkItDownWrapper",
        8103,
    )
    .with_upstream("http://markitdown:8005", Some("/health"))
    .with_tags(&["markdown", "documents"])
    .apply_env();

    run(config).await
}
