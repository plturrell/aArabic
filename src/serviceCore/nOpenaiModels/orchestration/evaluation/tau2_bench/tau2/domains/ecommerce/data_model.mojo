# Ecommerce Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the ecommerce domain

from collections import Dict, List

# Order status constants
alias ORDER_STATUS_PENDING = "pending"
alias ORDER_STATUS_PROCESSING = "processing"
alias ORDER_STATUS_SHIPPED = "shipped"
alias ORDER_STATUS_DELIVERED = "delivered"
alias ORDER_STATUS_CANCELLED = "cancelled"
alias ORDER_STATUS_RETURNED = "returned"

# Payment status constants
alias PAYMENT_STATUS_PENDING = "pending"
alias PAYMENT_STATUS_COMPLETED = "completed"
alias PAYMENT_STATUS_REFUNDED = "refunded"


struct Address:
    """Represents a shipping/billing address."""
    var address1: String
    var address2: String
    var city: String
    var state: String
    var country: String
    var zip: String
    
    fn __init__(out self, address1: String, city: String, state: String, 
                country: String, zip: String):
        self.address1 = address1
        self.address2 = ""
        self.city = city
        self.state = state
        self.country = country
        self.zip = zip


struct Product:
    """Represents a product."""
    var product_id: String
    var name: String
    var description: String
    var price: Float64
    var category: String
    var stock: Int
    var rating: Float64
    
    fn __init__(out self, product_id: String, name: String, price: Float64):
        self.product_id = product_id
        self.name = name
        self.description = ""
        self.price = price
        self.category = ""
        self.stock = 0
        self.rating = 0.0
    
    fn is_in_stock(self) -> Bool:
        return self.stock > 0


struct OrderItem:
    """Represents an item in an order."""
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
    """Represents an order."""
    var order_id: String
    var user_id: String
    var items: List[OrderItem]
    var status: String
    var shipping_address: Address
    var payment_status: String
    var created_at: String
    var total: Float64
    
    fn __init__(out self, order_id: String, user_id: String):
        self.order_id = order_id
        self.user_id = user_id
        self.items = List[OrderItem]()
        self.status = ORDER_STATUS_PENDING
        self.shipping_address = Address("", "", "", "", "")
        self.payment_status = PAYMENT_STATUS_PENDING
        self.created_at = ""
        self.total = 0.0


struct User:
    """Represents an ecommerce user."""
    var user_id: String
    var first_name: String
    var last_name: String
    var email: String
    var phone: String
    var address: Address
    var orders: List[String]  # List of order IDs
    
    fn __init__(out self, user_id: String, first_name: String, last_name: String,
                email: String):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = ""
        self.address = Address("", "", "", "", "")
        self.orders = List[String]()
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct EcommerceDB:
    """Database for the ecommerce domain."""
    var users: Dict[String, User]
    var products: Dict[String, Product]
    var orders: Dict[String, Order]
    
    fn __init__(out self):
        self.users = Dict[String, User]()
        self.products = Dict[String, Product]()
        self.orders = Dict[String, Order]()
    
    fn add_user(inout self, user: User):
        self.users[user.user_id] = user
    
    fn add_product(inout self, product: Product):
        self.products[product.product_id] = product
    
    fn add_order(inout self, order: Order):
        self.orders[order.order_id] = order
    
    fn get_user(self, user_id: String) raises -> User:
        if user_id not in self.users:
            raise Error("User " + user_id + " not found")
        return self.users[user_id]
    
    fn get_product(self, product_id: String) raises -> Product:
        if product_id not in self.products:
            raise Error("Product " + product_id + " not found")
        return self.products[product_id]
    
    fn get_order(self, order_id: String) raises -> Order:
        if order_id not in self.orders:
            raise Error("Order " + order_id + " not found")
        return self.orders[order_id]

