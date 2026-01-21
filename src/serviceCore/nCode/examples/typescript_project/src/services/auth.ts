/**
 * Authentication service for user management
 */

import { User, IUser } from '../models/user';
import { validateEmail, validatePassword, hashPassword } from '../utils/helpers';

export interface IAuthCredentials {
  email: string;
  password: string;
}

export interface IAuthToken {
  token: string;
  expiresAt: Date;
  userId: string;
}

/**
 * Authentication service class
 */
export class AuthService {
  private users: Map<string, User>;
  private tokens: Map<string, IAuthToken>;

  constructor() {
    this.users = new Map();
    this.tokens = new Map();
  }

  /**
   * Register a new user
   */
  async register(email: string, username: string, password: string): Promise<User> {
    // Validate input
    if (!validateEmail(email)) {
      throw new Error('Invalid email format');
    }

    if (!validatePassword(password)) {
      throw new Error('Password must be at least 8 characters with uppercase, lowercase, and numbers');
    }

    // Check if user already exists
    if (this.findUserByEmail(email)) {
      throw new Error('User with this email already exists');
    }

    // Create new user
    const passwordHash = await hashPassword(password);
    const userData: IUser = {
      id: this.generateUserId(),
      email,
      username,
      passwordHash,
      createdAt: new Date()
    };

    const user = new User(userData);
    this.users.set(user.id, user);

    return user;
  }

  /**
   * Authenticate user with credentials
   */
  async login(credentials: IAuthCredentials): Promise<IAuthToken> {
    const user = this.findUserByEmail(credentials.email);
    
    if (!user) {
      throw new Error('Invalid credentials');
    }

    // Verify password
    const isValidPassword = await this.verifyPassword(
      credentials.password,
      user.passwordHash
    );

    if (!isValidPassword) {
      throw new Error('Invalid credentials');
    }

    // Update last login
    user.updateLastLogin();

    // Generate token
    const token = this.generateToken(user);
    this.tokens.set(token.token, token);

    return token;
  }

  /**
   * Logout user and invalidate token
   */
  logout(tokenString: string): void {
    this.tokens.delete(tokenString);
  }

  /**
   * Verify authentication token
   */
  verifyToken(tokenString: string): IAuthToken | null {
    const token = this.tokens.get(tokenString);
    
    if (!token) {
      return null;
    }

    // Check if token is expired
    if (token.expiresAt < new Date()) {
      this.tokens.delete(tokenString);
      return null;
    }

    return token;
  }

  /**
   * Get user by token
   */
  getUserByToken(tokenString: string): User | null {
    const token = this.verifyToken(tokenString);
    
    if (!token) {
      return null;
    }

    return this.users.get(token.userId) || null;
  }

  /**
   * Change user password
   */
  async changePassword(userId: string, oldPassword: string, newPassword: string): Promise<void> {
    const user = this.users.get(userId);
    
    if (!user) {
      throw new Error('User not found');
    }

    // Verify old password
    const isValidPassword = await this.verifyPassword(oldPassword, user.passwordHash);
    
    if (!isValidPassword) {
      throw new Error('Invalid current password');
    }

    // Validate new password
    if (!validatePassword(newPassword)) {
      throw new Error('New password does not meet requirements');
    }

    // Update password
    user.passwordHash = await hashPassword(newPassword);
  }

  /**
   * Request password reset
   */
  async requestPasswordReset(email: string): Promise<string> {
    const user = this.findUserByEmail(email);
    
    if (!user) {
      throw new Error('User not found');
    }

    // Generate reset token
    const resetToken = this.generateResetToken();
    
    // In production, this would send an email
    console.log(`Password reset requested for ${email}`);
    
    return resetToken;
  }

  /**
   * Find user by email
   */
  private findUserByEmail(email: string): User | undefined {
    return Array.from(this.users.values()).find(u => u.email === email);
  }

  /**
   * Verify password against hash
   */
  private async verifyPassword(password: string, hash: string): Promise<boolean> {
    // In production, use bcrypt or similar
    const testHash = await hashPassword(password);
    return testHash === hash;
  }

  /**
   * Generate authentication token
   */
  private generateToken(user: User): IAuthToken {
    const token = this.randomString(64);
    const expiresAt = new Date();
    expiresAt.setHours(expiresAt.getHours() + 24); // 24 hour expiry

    return {
      token,
      expiresAt,
      userId: user.id
    };
  }

  /**
   * Generate unique user ID
   */
  private generateUserId(): string {
    return `user_${Date.now()}_${this.randomString(8)}`;
  }

  /**
   * Generate password reset token
   */
  private generateResetToken(): string {
    return this.randomString(32);
  }

  /**
   * Generate random string
   */
  private randomString(length: number): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  }

  /**
   * Get all users (admin only)
   */
  getAllUsers(): User[] {
    return Array.from(this.users.values());
  }

  /**
   * Get user by ID
   */
  getUserById(id: string): User | undefined {
    return this.users.get(id);
  }

  /**
   * Delete user
   */
  deleteUser(id: string): boolean {
    return this.users.delete(id);
  }
}
