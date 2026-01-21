# Railway Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the railway domain

from collections import Dict, List

# Ticket class constants
alias TICKET_CLASS_FIRST = "first"
alias TICKET_CLASS_BUSINESS = "business"
alias TICKET_CLASS_ECONOMY = "economy"

# Reservation status constants
alias RESERVATION_STATUS_ACTIVE = "active"
alias RESERVATION_STATUS_CANCELLED = "cancelled"
alias RESERVATION_STATUS_COMPLETED = "completed"


struct Station:
    """Represents a railway station."""
    var station_id: String
    var name: String
    var city: String
    var state: String
    var country: String
    
    fn __init__(out self, station_id: String, name: String, city: String):
        self.station_id = station_id
        self.name = name
        self.city = city
        self.state = ""
        self.country = ""


struct Train:
    """Represents a train."""
    var train_id: String
    var name: String
    var train_type: String
    var capacity: Int
    
    fn __init__(out self, train_id: String, name: String, train_type: String):
        self.train_id = train_id
        self.name = name
        self.train_type = train_type
        self.capacity = 200


struct Schedule:
    """Represents a train schedule."""
    var schedule_id: String
    var train_id: String
    var origin_station: String
    var destination_station: String
    var departure_time: String
    var arrival_time: String
    var date: String
    var available_seats: Int
    var price_first: Float64
    var price_business: Float64
    var price_economy: Float64
    
    fn __init__(out self, schedule_id: String, train_id: String, 
                origin: String, destination: String, departure: String,
                arrival: String, date: String):
        self.schedule_id = schedule_id
        self.train_id = train_id
        self.origin_station = origin
        self.destination_station = destination
        self.departure_time = departure
        self.arrival_time = arrival
        self.date = date
        self.available_seats = 100
        self.price_first = 0.0
        self.price_business = 0.0
        self.price_economy = 0.0


struct Passenger:
    """Represents a passenger."""
    var first_name: String
    var last_name: String
    var dob: String
    var passport_number: String
    
    fn __init__(out self, first_name: String, last_name: String, dob: String):
        self.first_name = first_name
        self.last_name = last_name
        self.dob = dob
        self.passport_number = ""


struct Reservation:
    """Represents a railway reservation."""
    var reservation_id: String
    var user_id: String
    var schedule_id: String
    var passengers: List[Passenger]
    var ticket_class: String
    var total_price: Float64
    var status: String
    var created_at: String
    
    fn __init__(out self, reservation_id: String, user_id: String, 
                schedule_id: String, ticket_class: String):
        self.reservation_id = reservation_id
        self.user_id = user_id
        self.schedule_id = schedule_id
        self.passengers = List[Passenger]()
        self.ticket_class = ticket_class
        self.total_price = 0.0
        self.status = RESERVATION_STATUS_ACTIVE
        self.created_at = ""


struct User:
    """Represents a railway domain user."""
    var user_id: String
    var first_name: String
    var last_name: String
    var email: String
    var phone: String
    var reservations: List[String]
    
    fn __init__(out self, user_id: String, first_name: String, last_name: String,
                email: String):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = ""
        self.reservations = List[String]()


struct RailwayDB:
    """Database for the railway domain."""
    var stations: Dict[String, Station]
    var trains: Dict[String, Train]
    var schedules: Dict[String, Schedule]
    var reservations: Dict[String, Reservation]
    var users: Dict[String, User]
    
    fn __init__(out self):
        self.stations = Dict[String, Station]()
        self.trains = Dict[String, Train]()
        self.schedules = Dict[String, Schedule]()
        self.reservations = Dict[String, Reservation]()
        self.users = Dict[String, User]()
    
    fn add_station(inout self, station: Station):
        self.stations[station.station_id] = station
    
    fn add_train(inout self, train: Train):
        self.trains[train.train_id] = train
    
    fn get_station(self, station_id: String) raises -> Station:
        if station_id not in self.stations:
            raise Error("Station " + station_id + " not found")
        return self.stations[station_id]

