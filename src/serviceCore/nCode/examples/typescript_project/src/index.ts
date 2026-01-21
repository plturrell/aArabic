/**
 * Main entry point for nCode TypeScript example
 */

import { User, UserProfile } from './models/user';
import { Product, ProductStatus } from './models/product';
import { AuthService } from './services/auth';
import { DatabaseConnection, BaseRepository } from './services/database';
import { formatCurrency, capitalize } from './utils/helpers';

/**
 * User repository implementation
 */
class UserRepository extends BaseRepository<User> {
  constructor(db: DatabaseConnection) {
    super(db, 'users');
  }

  async findByEmail(email: string): Promise<User | null> {
    const result = await this.db.query<User>(
      `SELECT * FROM ${this.tableName} WHERE email = $1`,
      [email]
    );
    return result.rows[0] || null;
  }
}

/**
 * Product repository implementation
 */
class ProductRepository extends BaseRepository<Product> {
  constructor(db: DatabaseConnection) {
    super(db, 'products');
  }

  async findByCategory(category: string): Promise<Product[]> {
    const result = await this.db.query<Product>(
      `SELECT * FROM ${this.tableName} WHERE category = $1`,
      [category]
    );
    return result.rows;
  }

  async findLowStock(threshold: number = 10): Promise<Product[]> {
    const result = await this.db.query<Product>(
      `SELECT * FROM ${this.tableName} WHERE stock > 0 AND stock <= $1`,
      [threshold]
    );
    return result.rows;
  }
}

/**
 * Application main class
 */
class Application {
  private db: DatabaseConnection;
  private authService: AuthService;
  private userRepository: UserRepository;
  private productRepository: ProductRepository;

  constructor() {
    this.db = new DatabaseConnection({
      host: 'localhost',
      port: 5432,
      database: 'example_db',
      username: 'user',
      password: 'password'
    });

    this.authService = new AuthService();
    this.userRepository = new UserRepository(this.db);
    this.productRepository = new ProductRepository(this.db);
  }

  /**
   * Initialize the application
   */
  async initialize(): Promise<void> {
    console.log('🚀 Initializing nCode TypeScript Example...');

    // Connect to database
    try {
      await this.db.connect();
      console.log('✅ Database connected');
    } catch (error) {
      console.error('❌ Database connection failed:', error);
      throw error;
    }

    // Check database ping
    const isAlive = await this.db.ping();
    console.log(`✅ Database ping: ${isAlive ? 'OK' : 'FAILED'}`);
  }

  /**
   * Run example operations
   */
  async runExamples(): Promise<void> {
    console.log('\n📝 Running example operations...\n');

    // Example 1: User registration and authentication
    await this.exampleUserAuth();

    // Example 2: Product management
    await this.exampleProductManagement();

    // Example 3: Database operations
    await this.exampleDatabaseOperations();
  }

  /**
   * Example: User authentication
   */
  private async exampleUserAuth(): Promise<void> {
    console.log('Example 1: User Registration & Authentication');
    console.log('─────────────────────────────────────────────');

    try {
      // Register a new user
      const user = await this.authService.register(
        'john.doe@example.com',
        'johndoe',
        'SecurePass123'
      );
      console.log(`✅ User registered: ${user.getDisplayName()}`);

      // Login
      const token = await this.authService.login({
        email: 'john.doe@example.com',
        password: 'SecurePass123'
      });
      console.log(`✅ User logged in, token expires: ${token.expiresAt.toISOString()}`);

      // Verify token
      const isValid = this.authService.verifyToken(token.token);
      console.log(`✅ Token valid: ${!!isValid}`);

      // Get user by token
      const authenticatedUser = this.authService.getUserByToken(token.token);
      console.log(`✅ Authenticated user: ${authenticatedUser?.getDisplayName()}`);

    } catch (error) {
      console.error('❌ Auth error:', error);
    }

    console.log('');
  }

  /**
   * Example: Product management
   */
  private async exampleProductManagement(): Promise<void> {
    console.log('Example 2: Product Management');
    console.log('─────────────────────────────────────────────');

    try {
      // Create a product
      const product = new Product({
        id: 'prod_001',
        name: 'Laptop Pro 15"',
        description: 'High-performance laptop for professionals',
        price: 1299.99,
        currency: 'USD',
        stock: 50,
        category: 'Electronics',
        tags: ['laptop', 'computer', 'tech'],
        imageUrls: ['https://example.com/laptop.jpg'],
        createdAt: new Date(),
        updatedAt: new Date()
      });

      console.log(`✅ Product created: ${product.name}`);
      console.log(`   Price: ${formatCurrency(product.price, product.currency)}`);
      console.log(`   Stock: ${product.stock} units`);
      console.log(`   Status: ${capitalize(product.getStatus())}`);

      // Reserve stock
      const reserved = product.reserveStock(5);
      console.log(`✅ Stock reserved: ${reserved ? 'Yes' : 'No'}`);
      console.log(`   Remaining stock: ${product.stock} units`);

      // Calculate discounted price
      const discountedPrice = product.calculateDiscountedPrice(20);
      console.log(`✅ 20% discount: ${formatCurrency(discountedPrice, product.currency)}`);

      // Check low stock
      const isLowStock = product.isLowStock(100);
      console.log(`✅ Low stock (threshold 100): ${isLowStock ? 'Yes' : 'No'}`);

    } catch (error) {
      console.error('❌ Product error:', error);
    }

    console.log('');
  }

  /**
   * Example: Database operations
   */
  private async exampleDatabaseOperations(): Promise<void> {
    console.log('Example 3: Database Operations');
    console.log('─────────────────────────────────────────────');

    try {
      // Get pool status
      const poolStatus = this.db.getPoolStatus();
      console.log(`✅ Connection pool: ${poolStatus.active}/${poolStatus.max} active`);

      // Database config
      const config = this.db.getConfig();
      console.log(`✅ Database config: ${config.host}:${config.port}/${config.database}`);

      // Simulate transaction
      await this.db.transaction(async (query) => {
        await query('INSERT INTO logs (message) VALUES ($1)', ['Transaction test']);
        return true;
      });
      console.log('✅ Transaction completed successfully');

    } catch (error) {
      console.error('❌ Database error:', error);
    }

    console.log('');
  }

  /**
   * Shutdown the application
   */
  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down application...');

    await this.db.disconnect();
    console.log('✅ Database disconnected');
    console.log('✅ Shutdown complete');
  }
}

/**
 * Main execution
 */
async function main(): Promise<void> {
  const app = new Application();

  try {
    await app.initialize();
    await app.runExamples();
  } catch (error) {
    console.error('❌ Application error:', error);
    process.exit(1);
  } finally {
    await app.shutdown();
  }

  console.log('\n✅ nCode TypeScript example completed!');
  console.log('   Next steps:');
  console.log('   1. Run: npx @sourcegraph/scip-typescript index');
  console.log('   2. Load to nCode: curl -X POST http://localhost:18003/v1/index/load ...');
  console.log('   3. Export to Qdrant: python ../../scripts/load_to_databases.py index.scip --qdrant');
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { Application, UserRepository, ProductRepository };
