# Airline Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the airline domain

from collections import Dict, List

# Flight type constants
alias FLIGHT_TYPE_ROUND_TRIP = "round_trip"
alias FLIGHT_TYPE_ONE_WAY = "one_way"

# Cabin class constants
alias CABIN_CLASS_BUSINESS = "business"
alias CABIN_CLASS_ECONOMY = "economy"
alias CABIN_CLASS_BASIC_ECONOMY = "basic_economy"

# Insurance constants
alias INSURANCE_YES = "yes"
alias INSURANCE_NO = "no"

# Membership level constants
alias MEMBERSHIP_GOLD = "gold"
alias MEMBERSHIP_SILVER = "silver"
alias MEMBERSHIP_REGULAR = "regular"

# Flight status constants
alias STATUS_AVAILABLE = "available"
alias STATUS_ON_TIME = "on time"
alias STATUS_FLYING = "flying"
alias STATUS_LANDED = "landed"
alias STATUS_CANCELLED = "cancelled"
alias STATUS_DELAYED = "delayed"


struct AirportCode:
    """Represents an airport with IATA code."""
    var iata: String
    var city: String
    
    fn __init__(out self, iata: String, city: String):
        self.iata = iata
        self.city = city


struct Name:
    """Represents a person's name."""
    var first_name: String
    var last_name: String
    
    fn __init__(out self, first_name: String, last_name: String):
        self.first_name = first_name
        self.last_name = last_name
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct Address:
    """Represents a physical address."""
    var address1: String
    var address2: String
    var city: String
    var country: String
    var state: String
    var zip: String
    
    fn __init__(out self, address1: String, city: String, country: String, 
                state: String, zip: String):
        self.address1 = address1
        self.address2 = ""
        self.city = city
        self.country = country
        self.state = state
        self.zip = zip
    
    fn __init__(out self, address1: String, address2: String, city: String, 
                country: String, state: String, zip: String):
        self.address1 = address1
        self.address2 = address2
        self.city = city
        self.country = country
        self.state = state
        self.zip = zip


struct Payment:
    """Represents a payment."""
    var payment_id: String
    var amount: Int
    
    fn __init__(out self, payment_id: String, amount: Int):
        self.payment_id = payment_id
        self.amount = amount


struct PaymentMethod:
    """Represents a payment method."""
    var source: String  # "credit_card", "gift_card", "certificate"
    var id: String
    var brand: String  # For credit cards
    var last_four: String  # For credit cards
    var amount: Float64  # For gift cards/certificates
    
    fn __init__(out self, source: String, id: String):
        self.source = source
        self.id = id
        self.brand = ""
        self.last_four = ""
        self.amount = 0.0
    
    fn is_credit_card(self) -> Bool:
        return self.source == "credit_card"
    
    fn is_gift_card(self) -> Bool:
        return self.source == "gift_card"
    
    fn is_certificate(self) -> Bool:
        return self.source == "certificate"


struct Passenger:
    """Represents a passenger."""
    var first_name: String
    var last_name: String
    var dob: String  # Date of birth in YYYY-MM-DD format
    
    fn __init__(out self, first_name: String, last_name: String, dob: String):
        self.first_name = first_name
        self.last_name = last_name
        self.dob = dob


struct SeatPrices:
    """Prices for different cabin classes."""
    var business: Int
    var economy: Int
    var basic_economy: Int
    
    fn __init__(out self, business: Int, economy: Int, basic_economy: Int):
        self.business = business
        self.economy = economy
        self.basic_economy = basic_economy
    
    fn get_price(self, cabin: String) -> Int:
        if cabin == CABIN_CLASS_BUSINESS:
            return self.business
        elif cabin == CABIN_CLASS_ECONOMY:
            return self.economy
        else:
            return self.basic_economy


struct AvailableSeats:
    """Available seats for different cabin classes."""
    var business: Int
    var economy: Int
    var basic_economy: Int

    fn __init__(out self, business: Int, economy: Int, basic_economy: Int):
        self.business = business
        self.economy = economy
        self.basic_economy = basic_economy

    fn get_seats(self, cabin: String) -> Int:
        if cabin == CABIN_CLASS_BUSINESS:
            return self.business
        elif cabin == CABIN_CLASS_ECONOMY:
            return self.economy
        else:
            return self.basic_economy


struct Flight:
    """Represents a flight."""
    var flight_number: String
    var origin: String
    var destination: String
    var scheduled_departure_time_est: String
    var scheduled_arrival_time_est: String
    var status: String
    var prices: SeatPrices
    var available_seats: AvailableSeats

    fn __init__(out self, flight_number: String, origin: String, destination: String,
                departure: String, arrival: String, status: String,
                prices: SeatPrices, seats: AvailableSeats):
        self.flight_number = flight_number
        self.origin = origin
        self.destination = destination
        self.scheduled_departure_time_est = departure
        self.scheduled_arrival_time_est = arrival
        self.status = status
        self.prices = prices
        self.available_seats = seats


struct Reservation:
    """Represents a flight reservation."""
    var reservation_id: String
    var user_id: String
    var origin: String
    var destination: String
    var flight_type: String
    var cabin: String
    var flights: List[String]  # List of flight numbers
    var passengers: List[Passenger]
    var payment_history: List[Payment]
    var created_at: String
    var total_baggages: Int
    var nonfree_baggages: Int
    var insurance: String
    var status: String

    fn __init__(out self, reservation_id: String, user_id: String):
        self.reservation_id = reservation_id
        self.user_id = user_id
        self.origin = ""
        self.destination = ""
        self.flight_type = FLIGHT_TYPE_ONE_WAY
        self.cabin = CABIN_CLASS_ECONOMY
        self.flights = List[String]()
        self.passengers = List[Passenger]()
        self.payment_history = List[Payment]()
        self.created_at = ""
        self.total_baggages = 0
        self.nonfree_baggages = 0
        self.insurance = INSURANCE_NO
        self.status = "active"


struct User:
    """Represents a user in the airline system."""
    var user_id: String
    var name: Name
    var address: Address
    var email: String
    var dob: String
    var payment_methods: List[PaymentMethod]
    var reservations: List[String]
    var membership: String

    fn __init__(out self, user_id: String, name: Name, email: String, dob: String):
        self.user_id = user_id
        self.name = name
        self.address = Address("", "", "", "", "")
        self.email = email
        self.dob = dob
        self.payment_methods = List[PaymentMethod]()
        self.reservations = List[String]()
        self.membership = MEMBERSHIP_REGULAR


struct AirlineDB:
    """Database for the airline domain."""
    var users: Dict[String, User]
    var reservations: Dict[String, Reservation]
    var flights: Dict[String, Flight]

    fn __init__(out self):
        self.users = Dict[String, User]()
        self.reservations = Dict[String, Reservation]()
        self.flights = Dict[String, Flight]()

