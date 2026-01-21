/**
 * Database connection and query service
 */

import { retry } from '../utils/helpers';

export interface IDatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  maxConnections?: number;
  connectionTimeout?: number;
}

export interface IQueryResult<T = any> {
  rows: T[];
  rowCount: number;
  fields: string[];
}

/**
 * Database connection class
 */
export class DatabaseConnection {
  private config: IDatabaseConfig;
  private connected: boolean = false;
  private connectionPool: any[] = [];

  constructor(config: IDatabaseConfig) {
    this.config = {
      ...config,
      maxConnections: config.maxConnections || 10,
      connectionTimeout: config.connectionTimeout || 5000
    };
  }

  /**
   * Connect to database
   */
  async connect(): Promise<void> {
    if (this.connected) {
      return;
    }

    try {
      await retry(async () => {
        // Simulate connection
        await this.simulateConnection();
        this.connected = true;
      }, 3, 1000);
    } catch (error) {
      throw new Error(`Failed to connect to database: ${error}`);
    }
  }

  /**
   * Disconnect from database
   */
  async disconnect(): Promise<void> {
    if (!this.connected) {
      return;
    }

    // Close all connections in pool
    this.connectionPool = [];
    this.connected = false;
  }

  /**
   * Execute a query
   */
  async query<T = any>(sql: string, params: any[] = []): Promise<IQueryResult<T>> {
    if (!this.connected) {
      throw new Error('Database not connected');
    }

    // Simulate query execution
    const result = await this.simulateQuery<T>(sql, params);
    return result;
  }

  /**
   * Execute a transaction
   */
  async transaction<T>(
    callback: (query: (sql: string, params?: any[]) => Promise<IQueryResult>) => Promise<T>
  ): Promise<T> {
    if (!this.connected) {
      throw new Error('Database not connected');
    }

    try {
      // Begin transaction
      await this.query('BEGIN');

      // Execute callback with query function
      const result = await callback(this.query.bind(this));

      // Commit transaction
      await this.query('COMMIT');

      return result;
    } catch (error) {
      // Rollback on error
      await this.query('ROLLBACK');
      throw error;
    }
  }

  /**
   * Check if database is connected
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * Get database configuration (without password)
   */
  getConfig(): Omit<IDatabaseConfig, 'password'> {
    const { password, ...safeConfig } = this.config;
    return safeConfig;
  }

  /**
   * Ping database to check connection
   */
  async ping(): Promise<boolean> {
    try {
      await this.query('SELECT 1');
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get connection pool status
   */
  getPoolStatus(): { active: number; max: number } {
    return {
      active: this.connectionPool.length,
      max: this.config.maxConnections || 10
    };
  }

  /**
   * Simulate database connection
   */
  private async simulateConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (Math.random() > 0.1) { // 90% success rate
          resolve();
        } else {
          reject(new Error('Connection timeout'));
        }
      }, 100);
    });
  }

  /**
   * Simulate query execution
   */
  private async simulateQuery<T>(sql: string, params: any[]): Promise<IQueryResult<T>> {
    return new Promise((resolve) => {
      setTimeout(() => {
        // Parse simple queries for demonstration
        const isSelect = sql.trim().toUpperCase().startsWith('SELECT');
        
        if (isSelect) {
          resolve({
            rows: [] as T[],
            rowCount: 0,
            fields: ['id', 'name', 'value']
          });
        } else {
          resolve({
            rows: [] as T[],
            rowCount: 1,
            fields: []
          });
        }
      }, 10);
    });
  }
}

/**
 * Database repository base class
 */
export abstract class BaseRepository<T> {
  protected db: DatabaseConnection;
  protected tableName: string;

  constructor(db: DatabaseConnection, tableName: string) {
    this.db = db;
    this.tableName = tableName;
  }

  /**
   * Find all records
   */
  async findAll(): Promise<T[]> {
    const result = await this.db.query<T>(
      `SELECT * FROM ${this.tableName}`
    );
    return result.rows;
  }

  /**
   * Find record by ID
   */
  async findById(id: string): Promise<T | null> {
    const result = await this.db.query<T>(
      `SELECT * FROM ${this.tableName} WHERE id = $1`,
      [id]
    );
    return result.rows[0] || null;
  }

  /**
   * Create a new record
   */
  async create(data: Partial<T>): Promise<T> {
    const keys = Object.keys(data);
    const values = Object.values(data);
    const placeholders = keys.map((_, i) => `$${i + 1}`).join(', ');

    const result = await this.db.query<T>(
      `INSERT INTO ${this.tableName} (${keys.join(', ')}) VALUES (${placeholders}) RETURNING *`,
      values
    );

    return result.rows[0];
  }

  /**
   * Update a record
   */
  async update(id: string, data: Partial<T>): Promise<T | null> {
    const keys = Object.keys(data);
    const values = Object.values(data);
    const setClause = keys.map((key, i) => `${key} = $${i + 1}`).join(', ');

    const result = await this.db.query<T>(
      `UPDATE ${this.tableName} SET ${setClause} WHERE id = $${keys.length + 1} RETURNING *`,
      [...values, id]
    );

    return result.rows[0] || null;
  }

  /**
   * Delete a record
   */
  async delete(id: string): Promise<boolean> {
    const result = await this.db.query(
      `DELETE FROM ${this.tableName} WHERE id = $1`,
      [id]
    );

    return result.rowCount > 0;
  }

  /**
   * Count records
   */
  async count(): Promise<number> {
    const result = await this.db.query<{ count: number }>(
      `SELECT COUNT(*) as count FROM ${this.tableName}`
    );

    return result.rows[0]?.count || 0;
  }

  /**
   * Check if record exists
   */
  async exists(id: string): Promise<boolean> {
    const result = await this.db.query<{ exists: boolean }>(
      `SELECT EXISTS(SELECT 1 FROM ${this.tableName} WHERE id = $1) as exists`,
      [id]
    );

    return result.rows[0]?.exists || false;
  }
}
