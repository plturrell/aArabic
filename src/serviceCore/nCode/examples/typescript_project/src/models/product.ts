/**
 * Product model for e-commerce system
 */

export interface IProduct {
  id: string;
  name: string;
  description: string;
  price: number;
  currency: string;
  stock: number;
  category: string;
  tags: string[];
  imageUrls: string[];
  createdAt: Date;
  updatedAt: Date;
}

export enum ProductStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  OUT_OF_STOCK = 'out_of_stock',
  DISCONTINUED = 'discontinued'
}

/**
 * Product class with inventory management
 */
export class Product implements IProduct {
  public id: string;
  public name: string;
  public description: string;
  public price: number;
  public currency: string;
  public stock: number;
  public category: string;
  public tags: string[];
  public imageUrls: string[];
  public createdAt: Date;
  public updatedAt: Date;
  private status: ProductStatus;

  constructor(data: IProduct) {
    this.id = data.id;
    this.name = data.name;
    this.description = data.description;
    this.price = data.price;
    this.currency = data.currency;
    this.stock = data.stock;
    this.category = data.category;
    this.tags = data.tags;
    this.imageUrls = data.imageUrls;
    this.createdAt = data.createdAt;
    this.updatedAt = data.updatedAt;
    this.status = this.calculateStatus();
  }

  /**
   * Calculate product status based on stock
   */
  private calculateStatus(): ProductStatus {
    if (this.stock === 0) return ProductStatus.OUT_OF_STOCK;
    return ProductStatus.ACTIVE;
  }

  /**
   * Get product status
   */
  getStatus(): ProductStatus {
    return this.status;
  }

  /**
   * Check if product is available for purchase
   */
  isAvailable(): boolean {
    return this.status === ProductStatus.ACTIVE && this.stock > 0;
  }

  /**
   * Update product stock
   */
  updateStock(quantity: number): void {
    this.stock += quantity;
    this.status = this.calculateStatus();
    this.updatedAt = new Date();
  }

  /**
   * Reserve stock for an order
   */
  reserveStock(quantity: number): boolean {
    if (quantity > this.stock) return false;
    this.stock -= quantity;
    this.status = this.calculateStatus();
    this.updatedAt = new Date();
    return true;
  }

  /**
   * Update product price
   */
  updatePrice(newPrice: number): void {
    if (newPrice < 0) throw new Error('Price cannot be negative');
    this.price = newPrice;
    this.updatedAt = new Date();
  }

  /**
   * Add tag to product
   */
  addTag(tag: string): void {
    if (!this.tags.includes(tag)) {
      this.tags.push(tag);
      this.updatedAt = new Date();
    }
  }

  /**
   * Remove tag from product
   */
  removeTag(tag: string): void {
    const index = this.tags.indexOf(tag);
    if (index > -1) {
      this.tags.splice(index, 1);
      this.updatedAt = new Date();
    }
  }

  /**
   * Calculate discounted price
   */
  calculateDiscountedPrice(discountPercent: number): number {
    if (discountPercent < 0 || discountPercent > 100) {
      throw new Error('Discount must be between 0 and 100');
    }
    return this.price * (1 - discountPercent / 100);
  }

  /**
   * Check if product is low on stock
   */
  isLowStock(threshold: number = 10): boolean {
    return this.stock > 0 && this.stock <= threshold;
  }

  /**
   * Convert to JSON-safe object
   */
  toJSON(): IProduct & { status: ProductStatus } {
    return {
      id: this.id,
      name: this.name,
      description: this.description,
      price: this.price,
      currency: this.currency,
      stock: this.stock,
      category: this.category,
      tags: this.tags,
      imageUrls: this.imageUrls,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      status: this.status
    };
  }
}
