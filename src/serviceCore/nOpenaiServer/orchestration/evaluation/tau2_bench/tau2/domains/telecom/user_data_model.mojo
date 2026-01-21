# Telecom User Data Model - Pure Mojo Implementation
# Defines user-specific data structures for the telecom domain

from collections import Dict, List


struct UserProfile:
    """Represents a user's profile in the telecom system."""
    var user_id: String
    var first_name: String
    var last_name: String
    var email: String
    var phone: String
    var date_of_birth: String
    var ssn_last_four: String
    var address: String
    var city: String
    var state: String
    var zip: String
    var country: String
    
    fn __init__(out self, user_id: String, first_name: String, last_name: String,
                email: String):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = ""
        self.date_of_birth = ""
        self.ssn_last_four = ""
        self.address = ""
        self.city = ""
        self.state = ""
        self.zip = ""
        self.country = ""
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct PaymentMethod:
    """Represents a payment method."""
    var payment_id: String
    var user_id: String
    var method_type: String  # "credit_card", "bank_account", "autopay"
    var last_four: String
    var is_default: Bool
    var expiry_date: String
    
    fn __init__(out self, payment_id: String, user_id: String, method_type: String):
        self.payment_id = payment_id
        self.user_id = user_id
        self.method_type = method_type
        self.last_four = ""
        self.is_default = False
        self.expiry_date = ""


struct UsageRecord:
    """Represents a usage record."""
    var record_id: String
    var account_id: String
    var usage_type: String  # "data", "voice", "text"
    var amount: Float64
    var timestamp: String
    var description: String
    
    fn __init__(out self, record_id: String, account_id: String, usage_type: String,
                amount: Float64, timestamp: String):
        self.record_id = record_id
        self.account_id = account_id
        self.usage_type = usage_type
        self.amount = amount
        self.timestamp = timestamp
        self.description = ""


struct SupportTicket:
    """Represents a support ticket."""
    var ticket_id: String
    var user_id: String
    var subject: String
    var description: String
    var status: String  # "open", "in_progress", "resolved", "closed"
    var priority: String  # "low", "medium", "high"
    var created_at: String
    var updated_at: String
    
    fn __init__(out self, ticket_id: String, user_id: String, subject: String):
        self.ticket_id = ticket_id
        self.user_id = user_id
        self.subject = subject
        self.description = ""
        self.status = "open"
        self.priority = "medium"
        self.created_at = ""
        self.updated_at = ""


struct UserDataDB:
    """Database for user-specific telecom data."""
    var profiles: Dict[String, UserProfile]
    var payment_methods: Dict[String, PaymentMethod]
    var usage_records: Dict[String, UsageRecord]
    var support_tickets: Dict[String, SupportTicket]
    
    fn __init__(out self):
        self.profiles = Dict[String, UserProfile]()
        self.payment_methods = Dict[String, PaymentMethod]()
        self.usage_records = Dict[String, UsageRecord]()
        self.support_tickets = Dict[String, SupportTicket]()
    
    fn add_profile(inout self, profile: UserProfile):
        self.profiles[profile.user_id] = profile
    
    fn add_payment_method(inout self, method: PaymentMethod):
        self.payment_methods[method.payment_id] = method
    
    fn add_support_ticket(inout self, ticket: SupportTicket):
        self.support_tickets[ticket.ticket_id] = ticket
    
    fn get_profile(self, user_id: String) raises -> UserProfile:
        if user_id not in self.profiles:
            raise Error("User profile " + user_id + " not found")
        return self.profiles[user_id]
    
    fn get_support_ticket(self, ticket_id: String) raises -> SupportTicket:
        if ticket_id not in self.support_tickets:
            raise Error("Support ticket " + ticket_id + " not found")
        return self.support_tickets[ticket_id]

