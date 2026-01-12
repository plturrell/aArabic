use service_common::{run, ServiceConfig};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config = ServiceConfig::new(
        "postgres",
        "Postgres",
        "data",
        "relational-db",
        "vendor/layerData/postgres",
        "src/serviceData/servicePostgres",
        8206,
    )
    .with_tags(&["sql", "postgres"])
    .apply_env();

    run(config).await
}
