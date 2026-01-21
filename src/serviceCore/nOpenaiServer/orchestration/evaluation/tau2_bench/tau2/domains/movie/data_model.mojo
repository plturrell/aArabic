# Movie Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the movie domain

from collections import Dict, List

# Booking status constants
alias BOOKING_STATUS_CONFIRMED = "confirmed"
alias BOOKING_STATUS_CANCELLED = "cancelled"
alias BOOKING_STATUS_PENDING = "pending"


struct Movie:
    """Represents a movie."""
    var movie_id: String
    var title: String
    var genre: String
    var duration: Int  # in minutes
    var rating: String  # e.g., "PG-13", "R"
    var release_date: String
    var description: String
    var average_rating: Float64
    
    fn __init__(out self, movie_id: String, title: String, genre: String,
                duration: Int, rating: String):
        self.movie_id = movie_id
        self.title = title
        self.genre = genre
        self.duration = duration
        self.rating = rating
        self.release_date = ""
        self.description = ""
        self.average_rating = 0.0


struct Theater:
    """Represents a movie theater."""
    var theater_id: String
    var name: String
    var address: String
    var city: String
    var state: String
    var zip: String
    var screens: Int
    
    fn __init__(out self, theater_id: String, name: String, address: String,
                city: String):
        self.theater_id = theater_id
        self.name = name
        self.address = address
        self.city = city
        self.state = ""
        self.zip = ""
        self.screens = 1


struct Showtime:
    """Represents a movie showtime."""
    var showtime_id: String
    var movie_id: String
    var theater_id: String
    var date: String
    var time: String
    var screen: Int
    var available_seats: Int
    var price: Float64
    
    fn __init__(out self, showtime_id: String, movie_id: String, theater_id: String,
                date: String, time: String, price: Float64):
        self.showtime_id = showtime_id
        self.movie_id = movie_id
        self.theater_id = theater_id
        self.date = date
        self.time = time
        self.screen = 1
        self.available_seats = 100
        self.price = price


struct Booking:
    """Represents a movie booking."""
    var booking_id: String
    var user_id: String
    var showtime_id: String
    var seats: List[String]
    var total_price: Float64
    var status: String
    var created_at: String
    
    fn __init__(out self, booking_id: String, user_id: String, showtime_id: String):
        self.booking_id = booking_id
        self.user_id = user_id
        self.showtime_id = showtime_id
        self.seats = List[String]()
        self.total_price = 0.0
        self.status = BOOKING_STATUS_PENDING
        self.created_at = ""


struct User:
    """Represents a movie domain user."""
    var user_id: String
    var first_name: String
    var last_name: String
    var email: String
    var phone: String
    var bookings: List[String]  # List of booking IDs
    
    fn __init__(out self, user_id: String, first_name: String, last_name: String,
                email: String):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = ""
        self.bookings = List[String]()
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct MovieDB:
    """Database for the movie domain."""
    var movies: Dict[String, Movie]
    var theaters: Dict[String, Theater]
    var showtimes: Dict[String, Showtime]
    var bookings: Dict[String, Booking]
    var users: Dict[String, User]
    
    fn __init__(out self):
        self.movies = Dict[String, Movie]()
        self.theaters = Dict[String, Theater]()
        self.showtimes = Dict[String, Showtime]()
        self.bookings = Dict[String, Booking]()
        self.users = Dict[String, User]()
    
    fn add_movie(inout self, movie: Movie):
        self.movies[movie.movie_id] = movie
    
    fn add_theater(inout self, theater: Theater):
        self.theaters[theater.theater_id] = theater
    
    fn add_showtime(inout self, showtime: Showtime):
        self.showtimes[showtime.showtime_id] = showtime
    
    fn get_movie(self, movie_id: String) raises -> Movie:
        if movie_id not in self.movies:
            raise Error("Movie " + movie_id + " not found")
        return self.movies[movie_id]
    
    fn get_theater(self, theater_id: String) raises -> Theater:
        if theater_id not in self.theaters:
            raise Error("Theater " + theater_id + " not found")
        return self.theaters[theater_id]

