# Restaurant Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the restaurant domain

from collections import Dict, List

# Reservation status constants
alias RESERVATION_STATUS_CONFIRMED = "confirmed"
alias RESERVATION_STATUS_CANCELLED = "cancelled"
alias RESERVATION_STATUS_COMPLETED = "completed"
alias RESERVATION_STATUS_NO_SHOW = "no_show"


struct Restaurant:
    """Represents a restaurant."""
    var restaurant_id: String
    var name: String
    var cuisine: String
    var address: String
    var city: String
    var state: String
    var zip: String
    var phone: String
    var rating: Float64
    var price_range: String  # "$", "$$", "$$$", "$$$$"
    var capacity: Int
    
    fn __init__(out self, restaurant_id: String, name: String, cuisine: String,
                address: String, city: String):
        self.restaurant_id = restaurant_id
        self.name = name
        self.cuisine = cuisine
        self.address = address
        self.city = city
        self.state = ""
        self.zip = ""
        self.phone = ""
        self.rating = 0.0
        self.price_range = "$$"
        self.capacity = 50


struct TimeSlot:
    """Represents an available time slot."""
    var time: String
    var available_tables: Int
    
    fn __init__(out self, time: String, available_tables: Int):
        self.time = time
        self.available_tables = available_tables


struct Reservation:
    """Represents a restaurant reservation."""
    var reservation_id: String
    var user_id: String
    var restaurant_id: String
    var date: String
    var time: String
    var party_size: Int
    var special_requests: String
    var status: String
    var created_at: String
    
    fn __init__(out self, reservation_id: String, user_id: String,
                restaurant_id: String, date: String, time: String, party_size: Int):
        self.reservation_id = reservation_id
        self.user_id = user_id
        self.restaurant_id = restaurant_id
        self.date = date
        self.time = time
        self.party_size = party_size
        self.special_requests = ""
        self.status = RESERVATION_STATUS_CONFIRMED
        self.created_at = ""


struct User:
    """Represents a restaurant domain user."""
    var user_id: String
    var first_name: String
    var last_name: String
    var email: String
    var phone: String
    var reservations: List[String]
    var favorite_restaurants: List[String]
    
    fn __init__(out self, user_id: String, first_name: String, last_name: String,
                email: String):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = ""
        self.reservations = List[String]()
        self.favorite_restaurants = List[String]()
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct RestaurantDB:
    """Database for the restaurant domain."""
    var restaurants: Dict[String, Restaurant]
    var reservations: Dict[String, Reservation]
    var users: Dict[String, User]
    
    fn __init__(out self):
        self.restaurants = Dict[String, Restaurant]()
        self.reservations = Dict[String, Reservation]()
        self.users = Dict[String, User]()
    
    fn add_restaurant(inout self, restaurant: Restaurant):
        self.restaurants[restaurant.restaurant_id] = restaurant
    
    fn add_reservation(inout self, reservation: Reservation):
        self.reservations[reservation.reservation_id] = reservation
    
    fn add_user(inout self, user: User):
        self.users[user.user_id] = user
    
    fn get_restaurant(self, restaurant_id: String) raises -> Restaurant:
        if restaurant_id not in self.restaurants:
            raise Error("Restaurant " + restaurant_id + " not found")
        return self.restaurants[restaurant_id]
    
    fn get_reservation(self, reservation_id: String) raises -> Reservation:
        if reservation_id not in self.reservations:
            raise Error("Reservation " + reservation_id + " not found")
        return self.reservations[reservation_id]
    
    fn get_user(self, user_id: String) raises -> User:
        if user_id not in self.users:
            raise Error("User " + user_id + " not found")
        return self.users[user_id]

