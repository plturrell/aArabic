use anyhow::Result;
use postgres::{Client, NoTls, Row};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete PostgreSQL API Client
pub struct PostgresClient {
    connection_string: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfo {
    pub table_name: String,
    pub table_schema: String,
    pub table_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub column_name: String,
    pub data_type: String,
    pub is_nullable: String,
    pub column_default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub index_name: String,
    pub table_name: String,
    pub column_name: String,
    pub is_unique: bool,
}

impl PostgresClient {
    /// Create new PostgreSQL client
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }

    /// Get connection
    fn connect(&self) -> Result<Client> {
        Ok(Client::connect(&self.connection_string, NoTls)?)
    }

    // ========================================================================
    // QUERY OPERATIONS
    // ========================================================================

    /// Execute query
    pub fn execute(&self, query: &str) -> Result<Vec<Row>> {
        let mut client = self.connect()?;
        let rows = client.query(query, &[])?;
        Ok(rows)
    }

    /// Execute query with parameters
    pub fn execute_params(&self, query: &str, params: &[&(dyn postgres::types::ToSql + Sync)]) -> Result<Vec<Row>> {
        let mut client = self.connect()?;
        let rows = client.query(query, params)?;
        Ok(rows)
    }

    /// Execute update/insert/delete
    pub fn execute_update(&self, query: &str) -> Result<u64> {
        let mut client = self.connect()?;
        let count = client.execute(query, &[])?;
        Ok(count)
    }

    // ========================================================================
    // DATABASE OPERATIONS
    // ========================================================================

    /// List databases
    pub fn list_databases(&self) -> Result<Vec<String>> {
        let mut client = self.connect()?;
        let rows = client.query("SELECT datname FROM pg_database WHERE datistemplate = false", &[])?;
        Ok(rows.iter().map(|row| row.get(0)).collect())
    }

    /// Create database
    pub fn create_database(&self, name: &str) -> Result<()> {
        let mut client = self.connect()?;
        client.execute(&format!("CREATE DATABASE {}", name), &[])?;
        Ok(())
    }

    /// Drop database
    pub fn drop_database(&self, name: &str) -> Result<()> {
        let mut client = self.connect()?;
        client.execute(&format!("DROP DATABASE {}", name), &[])?;
        Ok(())
    }

    /// Get current database
    pub fn current_database(&self) -> Result<String> {
        let mut client = self.connect()?;
        let row = client.query_one("SELECT current_database()", &[])?;
        Ok(row.get(0))
    }

    // ========================================================================
    // SCHEMA OPERATIONS
    // ========================================================================

    /// List schemas
    pub fn list_schemas(&self) -> Result<Vec<String>> {
        let mut client = self.connect()?;
        let rows = client.query(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog', 'information_schema')",
            &[],
        )?;
        Ok(rows.iter().map(|row| row.get(0)).collect())
    }

    /// Create schema
    pub fn create_schema(&self, name: &str) -> Result<()> {
        let mut client = self.connect()?;
        client.execute(&format!("CREATE SCHEMA {}", name), &[])?;
        Ok(())
    }

    /// Drop schema
    pub fn drop_schema(&self, name: &str, cascade: bool) -> Result<()> {
        let mut client = self.connect()?;
        let query = if cascade {
            format!("DROP SCHEMA {} CASCADE", name)
        } else {
            format!("DROP SCHEMA {}", name)
        };
        client.execute(&query, &[])?;
        Ok(())
    }

    // ========================================================================
    // TABLE OPERATIONS
    // ========================================================================

    /// List tables
    pub fn list_tables(&self, schema: Option<&str>) -> Result<Vec<TableInfo>> {
        let mut client = self.connect()?;
        let query = if let Some(s) = schema {
            format!(
                "SELECT table_name, table_schema, table_type FROM information_schema.tables WHERE table_schema = '{}'",
                s
            )
        } else {
            "SELECT table_name, table_schema, table_type FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')".to_string()
        };

        let rows = client.query(&query, &[])?;
        Ok(rows
            .iter()
            .map(|row| TableInfo {
                table_name: row.get(0),
                table_schema: row.get(1),
                table_type: row.get(2),
            })
            .collect())
    }

    /// Create table
    pub fn create_table(&self, name: &str, columns: &str) -> Result<()> {
        let mut client = self.connect()?;
        let query = format!("CREATE TABLE {} ({})", name, columns);
        client.execute(&query, &[])?;
        Ok(())
    }

    /// Drop table
    pub fn drop_table(&self, name: &str, cascade: bool) -> Result<()> {
        let mut client = self.connect()?;
        let query = if cascade {
            format!("DROP TABLE {} CASCADE", name)
        } else {
            format!("DROP TABLE {}", name)
        };
        client.execute(&query, &[])?;
        Ok(())
    }

    /// Describe table
    pub fn describe_table(&self, table: &str) -> Result<Vec<ColumnInfo>> {
        let mut client = self.connect()?;
        let rows = client.query(
            "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_name = $1",
            &[&table],
        )?;

        Ok(rows
            .iter()
            .map(|row| ColumnInfo {
                column_name: row.get(0),
                data_type: row.get(1),
                is_nullable: row.get(2),
                column_default: row.get(3),
            })
            .collect())
    }

    /// Truncate table
    pub fn truncate_table(&self, name: &str, cascade: bool) -> Result<()> {
        let mut client = self.connect()?;
        let query = if cascade {
            format!("TRUNCATE TABLE {} CASCADE", name)
        } else {
            format!("TRUNCATE TABLE {}", name)
        };
        client.execute(&query, &[])?;
        Ok(())
    }

    // ========================================================================
    // INDEX OPERATIONS
    // ========================================================================

    /// List indexes
    pub fn list_indexes(&self, table: Option<&str>) -> Result<Vec<IndexInfo>> {
        let mut client = self.connect()?;
        let query = if let Some(t) = table {
            format!(
                "SELECT indexname, tablename, indexdef FROM pg_indexes WHERE tablename = '{}'",
                t
            )
        } else {
            "SELECT indexname, tablename, indexdef FROM pg_indexes WHERE schemaname NOT IN ('pg_catalog', 'information_schema')".to_string()
        };

        let rows = client.query(&query, &[])?;
        Ok(rows
            .iter()
            .map(|row| IndexInfo {
                index_name: row.get(0),
                table_name: row.get(1),
                column_name: row.get::<_, String>(2),
                is_unique: false,
            })
            .collect())
    }

    /// Create index
    pub fn create_index(&self, name: &str, table: &str, column: &str, unique: bool) -> Result<()> {
        let mut client = self.connect()?;
        let query = if unique {
            format!("CREATE UNIQUE INDEX {} ON {} ({})", name, table, column)
        } else {
            format!("CREATE INDEX {} ON {} ({})", name, table, column)
        };
        client.execute(&query, &[])?;
        Ok(())
    }

    /// Drop index
    pub fn drop_index(&self, name: &str) -> Result<()> {
        let mut client = self.connect()?;
        client.execute(&format!("DROP INDEX {}", name), &[])?;
        Ok(())
    }

    // ========================================================================
    // USER OPERATIONS
    // ========================================================================

    /// List users
    pub fn list_users(&self) -> Result<Vec<String>> {
        let mut client = self.connect()?;
        let rows = client.query("SELECT usename FROM pg_user", &[])?;
        Ok(rows.iter().map(|row| row.get(0)).collect())
    }

    /// Create user
    pub fn create_user(&self, username: &str, password: &str) -> Result<()> {
        let mut client = self.connect()?;
        client.execute(
            &format!("CREATE USER {} WITH PASSWORD '{}'", username, password),
            &[],
        )?;
        Ok(())
    }

    /// Drop user
    pub fn drop_user(&self, username: &str) -> Result<()> {
        let mut client = self.connect()?;
        client.execute(&format!("DROP USER {}", username), &[])?;
        Ok(())
    }

    /// Grant privileges
    pub fn grant_privileges(&self, privilege: &str, table: &str, user: &str) -> Result<()> {
        let mut client = self.connect()?;
        client.execute(&format!("GRANT {} ON {} TO {}", privilege, table, user), &[])?;
        Ok(())
    }

    // ========================================================================
    // TRANSACTION OPERATIONS
    // ========================================================================

    /// Begin transaction
    pub fn begin_transaction(&self) -> Result<()> {
        let mut client = self.connect()?;
        client.execute("BEGIN", &[])?;
        Ok(())
    }

    /// Commit transaction
    pub fn commit_transaction(&self) -> Result<()> {
        let mut client = self.connect()?;
        client.execute("COMMIT", &[])?;
        Ok(())
    }

    /// Rollback transaction
    pub fn rollback_transaction(&self) -> Result<()> {
        let mut client = self.connect()?;
        client.execute("ROLLBACK", &[])?;
        Ok(())
    }

    // ========================================================================
    // BACKUP & RESTORE
    // ========================================================================

    /// Export table to CSV
    pub fn export_to_csv(&self, table: &str, file_path: &str) -> Result<()> {
        let mut client = self.connect()?;
        let query = format!("COPY {} TO '{}' CSV HEADER", table, file_path);
        client.execute(&query, &[])?;
        Ok(())
    }

    /// Import from CSV
    pub fn import_from_csv(&self, table: &str, file_path: &str) -> Result<()> {
        let mut client = self.connect()?;
        let query = format!("COPY {} FROM '{}' CSV HEADER", table, file_path);
        client.execute(&query, &[])?;
        Ok(())
    }

    // ========================================================================
    // STATISTICS
    // ========================================================================

    /// Get database size
    pub fn database_size(&self, name: &str) -> Result<String> {
        let mut client = self.connect()?;
        let row = client.query_one(
            "SELECT pg_size_pretty(pg_database_size($1))",
            &[&name],
        )?;
        Ok(row.get(0))
    }

    /// Get table size
    pub fn table_size(&self, name: &str) -> Result<String> {
        let mut client = self.connect()?;
        let row = client.query_one(
            "SELECT pg_size_pretty(pg_total_relation_size($1))",
            &[&name],
        )?;
        Ok(row.get(0))
    }

    /// Get row count
    pub fn row_count(&self, table: &str) -> Result<i64> {
        let mut client = self.connect()?;
        let row = client.query_one(&format!("SELECT COUNT(*) FROM {}", table), &[])?;
        Ok(row.get(0))
    }

    /// Get database info
    pub fn get_info(&self) -> Result<HashMap<String, String>> {
        let mut client = self.connect()?;
        let mut info = HashMap::new();

        let version: String = client.query_one("SELECT version()", &[])?.get(0);
        info.insert("version".to_string(), version);

        let current_db: String = client.query_one("SELECT current_database()", &[])?.get(0);
        info.insert("database".to_string(), current_db);

        let tables: i64 = client.query_one(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')",
            &[],
        )?.get(0);
        info.insert("tables".to_string(), tables.to_string());

        Ok(info)
    }

    // ========================================================================
    // ADVANCED OPERATIONS
    // ========================================================================

    /// Vacuum table
    pub fn vacuum(&self, table: Option<&str>, analyze: bool) -> Result<()> {
        let mut client = self.connect()?;
        let query = match (table, analyze) {
            (Some(t), true) => format!("VACUUM ANALYZE {}", t),
            (Some(t), false) => format!("VACUUM {}", t),
            (None, true) => "VACUUM ANALYZE".to_string(),
            (None, false) => "VACUUM".to_string(),
        };
        client.execute(&query, &[])?;
        Ok(())
    }

    /// Analyze table
    pub fn analyze(&self, table: Option<&str>) -> Result<()> {
        let mut client = self.connect()?;
        let query = if let Some(t) = table {
            format!("ANALYZE {}", t)
        } else {
            "ANALYZE".to_string()
        };
        client.execute(&query, &[])?;
        Ok(())
    }

    /// Reindex
    pub fn reindex(&self, target: &str) -> Result<()> {
        let mut client = self.connect()?;
        client.execute(&format!("REINDEX TABLE {}", target), &[])?;
        Ok(())
    }
}
