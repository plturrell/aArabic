use anyhow::Result;
use clap::{Parser, Subcommand};
use postgres_api_client::*;

#[derive(Parser)]
#[command(name = "postgres-cli")]
#[command(about = "PostgreSQL Database Client\n\nComplete database management")]
#[command(version = "1.0.0")]
struct Cli {
    #[arg(short, long, env = "DATABASE_URL")]
    connection: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // Query operations
    Query { sql: String },
    Execute { sql: String },
    
    // Database operations
    ListDatabases,
    CreateDatabase { name: String },
    DropDatabase { name: String },
    CurrentDatabase,
    DatabaseSize { name: String },
    Info,
    
    // Schema operations
    ListSchemas,
    CreateSchema { name: String },
    DropSchema { name: String, #[arg(long)] cascade: bool },
    
    // Table operations
    ListTables { #[arg(short, long)] schema: Option<String> },
    CreateTable { name: String, columns: String },
    DropTable { name: String, #[arg(long)] cascade: bool },
    DescribeTable { name: String },
    TruncateTable { name: String, #[arg(long)] cascade: bool },
    TableSize { name: String },
    RowCount { table: String },
    
    // Index operations
    ListIndexes { #[arg(short, long)] table: Option<String> },
    CreateIndex { name: String, table: String, column: String, #[arg(long)] unique: bool },
    DropIndex { name: String },
    
    // User operations
    ListUsers,
    CreateUser { username: String, password: String },
    DropUser { username: String },
    GrantPrivileges { privilege: String, table: String, user: String },
    
    // Transaction operations
    Begin,
    Commit,
    Rollback,
    
    // Backup & restore
    ExportCsv { table: String, file: String },
    ImportCsv { table: String, file: String },
    
    // Maintenance
    Vacuum { #[arg(short, long)] table: Option<String>, #[arg(long)] analyze: bool },
    Analyze { #[arg(short, long)] table: Option<String> },
    Reindex { target: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = PostgresClient::new(cli.connection);

    match cli.command {
        Commands::Query { sql } => {
            let rows = client.execute(&sql)?;
            println!("📊 Results ({} rows):", rows.len());
            for (i, row) in rows.iter().enumerate() {
                println!("   {}. {:?}", i + 1, row);
            }
        }

        Commands::Execute { sql } => {
            let count = client.execute_update(&sql)?;
            println!("✅ Affected {} rows", count);
        }

        Commands::ListDatabases => {
            let dbs = client.list_databases()?;
            println!("💾 Databases ({}):", dbs.len());
            for db in dbs {
                println!("   • {}", db);
            }
        }

        Commands::CreateDatabase { name } => {
            client.create_database(&name)?;
            println!("✅ Created database: {}", name);
        }

        Commands::DropDatabase { name } => {
            client.drop_database(&name)?;
            println!("✅ Dropped database: {}", name);
        }

        Commands::CurrentDatabase => {
            let db = client.current_database()?;
            println!("📍 Current database: {}", db);
        }

        Commands::DatabaseSize { name } => {
            let size = client.database_size(&name)?;
            println!("💾 Database {} size: {}", name, size);
        }

        Commands::Info => {
            let info = client.get_info()?;
            println!("📊 Database Info:");
            for (key, value) in info {
                println!("   {}: {}", key, value);
            }
        }

        Commands::ListSchemas => {
            let schemas = client.list_schemas()?;
            println!("📁 Schemas ({}):", schemas.len());
            for schema in schemas {
                println!("   • {}", schema);
            }
        }

        Commands::CreateSchema { name } => {
            client.create_schema(&name)?;
            println!("✅ Created schema: {}", name);
        }

        Commands::DropSchema { name, cascade } => {
            client.drop_schema(&name, cascade)?;
            println!("✅ Dropped schema: {}", name);
        }

        Commands::ListTables { schema } => {
            let tables = client.list_tables(schema.as_deref())?;
            println!("📊 Tables ({}):", tables.len());
            for table in tables {
                println!("   • {}.{} ({})", table.table_schema, table.table_name, table.table_type);
            }
        }

        Commands::CreateTable { name, columns } => {
            client.create_table(&name, &columns)?;
            println!("✅ Created table: {}", name);
        }

        Commands::DropTable { name, cascade } => {
            client.drop_table(&name, cascade)?;
            println!("✅ Dropped table: {}", name);
        }

        Commands::DescribeTable { name } => {
            let columns = client.describe_table(&name)?;
            println!("📋 Table: {}", name);
            for col in columns {
                println!("   • {} {} {}", 
                    col.column_name, 
                    col.data_type,
                    if col.is_nullable == "YES" { "NULL" } else { "NOT NULL" }
                );
            }
        }

        Commands::TruncateTable { name, cascade } => {
            client.truncate_table(&name, cascade)?;
            println!("✅ Truncated table: {}", name);
        }

        Commands::TableSize { name } => {
            let size = client.table_size(&name)?;
            println!("📊 Table {} size: {}", name, size);
        }

        Commands::RowCount { table } => {
            let count = client.row_count(&table)?;
            println!("📊 Table {} has {} rows", table, count);
        }

        Commands::ListIndexes { table } => {
            let indexes = client.list_indexes(table.as_deref())?;
            println!("🔍 Indexes ({}):", indexes.len());
            for idx in indexes {
                println!("   • {} on {}", idx.index_name, idx.table_name);
            }
        }

        Commands::CreateIndex { name, table, column, unique } => {
            client.create_index(&name, &table, &column, unique)?;
            println!("✅ Created index: {} on {}", name, table);
        }

        Commands::DropIndex { name } => {
            client.drop_index(&name)?;
            println!("✅ Dropped index: {}", name);
        }

        Commands::ListUsers => {
            let users = client.list_users()?;
            println!("👥 Users ({}):", users.len());
            for user in users {
                println!("   • {}", user);
            }
        }

        Commands::CreateUser { username, password } => {
            client.create_user(&username, &password)?;
            println!("✅ Created user: {}", username);
        }

        Commands::DropUser { username } => {
            client.drop_user(&username)?;
            println!("✅ Dropped user: {}", username);
        }

        Commands::GrantPrivileges { privilege, table, user } => {
            client.grant_privileges(&privilege, &table, &user)?;
            println!("✅ Granted {} on {} to {}", privilege, table, user);
        }

        Commands::Begin => {
            client.begin_transaction()?;
            println!("▶️  Transaction begun");
        }

        Commands::Commit => {
            client.commit_transaction()?;
            println!("✅ Transaction committed");
        }

        Commands::Rollback => {
            client.rollback_transaction()?;
            println!("↩️  Transaction rolled back");
        }

        Commands::ExportCsv { table, file } => {
            client.export_to_csv(&table, &file)?;
            println!("✅ Exported {} to {}", table, file);
        }

        Commands::ImportCsv { table, file } => {
            client.import_from_csv(&table, &file)?;
            println!("✅ Imported {} from {}", table, file);
        }

        Commands::Vacuum { table, analyze } => {
            client.vacuum(table.as_deref(), analyze)?;
            println!("✅ Vacuum completed");
        }

        Commands::Analyze { table } => {
            client.analyze(table.as_deref())?;
            println!("✅ Analyze completed");
        }

        Commands::Reindex { target } => {
            client.reindex(&target)?;
            println!("✅ Reindex completed for {}", target);
        }
    }

    Ok(())
}
