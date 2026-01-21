# Retail Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the retail domain

from collections import Dict, List

# Order status constants
alias ORDER_STATUS_PENDING = "pending"
alias ORDER_STATUS_PROCESSING = "processing"
alias ORDER_STATUS_SHIPPED = "shipped"
alias ORDER_STATUS_DELIVERED = "delivered"
alias ORDER_STATUS_CANCELLED = "cancelled"
alias ORDER_STATUS_RETURNED = "returned"


struct Product:
    """Represents a retail product."""
    var product_id: String
    var name: String
    var description: String
    var category: String
    var price: Float64
    var stock: Int
    var brand: String
    var rating: Float64
    
    fn __init__(out self, product_id: String, name: String, price: Float64):
        self.product_id = product_id
        self.name = name
        self.description = ""
        self.category = ""
        self.price = price
        self.stock = 0
        self.brand = ""
        self.rating = 0.0
    
    fn is_in_stock(self) -> Bool:
        return self.stock > 0


struct CartItem:
    """Represents an item in a shopping cart."""
    var product_id: String
    var quantity: Int
    var price: Float64
    
    fn __init__(out self, product_id: String, quantity: Int, price: Float64):
        self.product_id = product_id
        self.quantity = quantity
        self.price = price
    
    fn total(self) -> Float64:
        return self.price * Float64(self.quantity)


struct Order:
    """Represents a retail order."""
    var order_id: String
    var user_id: String
    var items: List[CartItem]
    var status: String
    var shipping_address: String
    var total: Float64
    var created_at: String
    var tracking_number: String
    
    fn __init__(out self, order_id: String, user_id: String):
        self.order_id = order_id
        self.user_id = user_id
        self.items = List[CartItem]()
        self.status = ORDER_STATUS_PENDING
        self.shipping_address = ""
        self.total = 0.0
        self.created_at = ""
        self.tracking_number = ""


struct User:
    """Represents a retail domain user."""
    var user_id: String
    var first_name: String
    var last_name: String
    var email: String
    var phone: String
    var address: String
    var orders: List[String]
    var cart: List[CartItem]
    
    fn __init__(out self, user_id: String, first_name: String, last_name: String,
                email: String):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = ""
        self.address = ""
        self.orders = List[String]()
        self.cart = List[CartItem]()
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct RetailDB:
    """Database for the retail domain."""
    var products: Dict[String, Product]
    var orders: Dict[String, Order]
    var users: Dict[String, User]
    
    fn __init__(out self):
        self.products = Dict[String, Product]()
        self.orders = Dict[String, Order]()
        self.users = Dict[String, User]()
    
    fn add_product(inout self, product: Product):
        self.products[product.product_id] = product
    
    fn add_order(inout self, order: Order):
        self.orders[order.order_id] = order
    
    fn add_user(inout self, user: User):
        self.users[user.user_id] = user
    
    fn get_product(self, product_id: String) raises -> Product:
        if product_id not in self.products:
            raise Error("Product " + product_id + " not found")
        return self.products[product_id]
    
    fn get_order(self, order_id: String) raises -> Order:
        if order_id not in self.orders:
            raise Error("Order " + order_id + " not found")
        return self.orders[order_id]
    
    fn get_user(self, user_id: String) raises -> User:
        if user_id not in self.users:
            raise Error("User " + user_id + " not found")
        return self.users[user_id]

