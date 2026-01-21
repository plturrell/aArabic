/**
 * User model and interface definitions
 */

export interface IUser {
  id: string;
  email: string;
  username: string;
  passwordHash: string;
  createdAt: Date;
  lastLoginAt?: Date;
}

export interface IUserProfile {
  userId: string;
  firstName: string;
  lastName: string;
  bio?: string;
  avatarUrl?: string;
}

/**
 * User class representing a system user
 */
export class User implements IUser {
  public id: string;
  public email: string;
  public username: string;
  public passwordHash: string;
  public createdAt: Date;
  public lastLoginAt?: Date;

  constructor(data: IUser) {
    this.id = data.id;
    this.email = data.email;
    this.username = data.username;
    this.passwordHash = data.passwordHash;
    this.createdAt = data.createdAt;
    this.lastLoginAt = data.lastLoginAt;
  }

  /**
   * Get the full display name of the user
   */
  getDisplayName(): string {
    return this.username || this.email;
  }

  /**
   * Check if user has logged in recently
   */
  hasRecentLogin(days: number = 7): boolean {
    if (!this.lastLoginAt) return false;
    const daysSinceLogin = (Date.now() - this.lastLoginAt.getTime()) / (1000 * 60 * 60 * 24);
    return daysSinceLogin <= days;
  }

  /**
   * Update last login timestamp
   */
  updateLastLogin(): void {
    this.lastLoginAt = new Date();
  }

  /**
   * Convert user to JSON-safe object
   */
  toJSON(): Omit<IUser, 'passwordHash'> {
    return {
      id: this.id,
      email: this.email,
      username: this.username,
      createdAt: this.createdAt,
      lastLoginAt: this.lastLoginAt
    };
  }
}

/**
 * User profile class with additional information
 */
export class UserProfile implements IUserProfile {
  public userId: string;
  public firstName: string;
  public lastName: string;
  public bio?: string;
  public avatarUrl?: string;

  constructor(data: IUserProfile) {
    this.userId = data.userId;
    this.firstName = data.firstName;
    this.lastName = data.lastName;
    this.bio = data.bio;
    this.avatarUrl = data.avatarUrl;
  }

  /**
   * Get full name of the user
   */
  getFullName(): string {
    return `${this.firstName} ${this.lastName}`;
  }

  /**
   * Check if profile is complete
   */
  isComplete(): boolean {
    return !!(this.firstName && this.lastName && this.bio && this.avatarUrl);
  }
}
