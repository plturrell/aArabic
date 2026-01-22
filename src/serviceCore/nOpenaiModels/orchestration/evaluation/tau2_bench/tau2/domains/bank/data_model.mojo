# Bank Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the bank domain

from collections import Dict, List

# Account type constants
alias ACCOUNT_TYPE_CHECKING = "checking"
alias ACCOUNT_TYPE_SAVINGS = "savings"
alias ACCOUNT_TYPE_CERTIFICATE_OF_DEPOSIT = "certificate_of_deposit"

# Transaction type constants
alias TRANSACTION_DEPOSIT = "deposit"
alias TRANSACTION_WITHDRAWAL = "withdrawal"
alias TRANSACTION_TRANSFER = "transfer"

# Account status constants
alias ACCOUNT_STATUS_ACTIVE = "active"
alias ACCOUNT_STATUS_CLOSED = "closed"


struct Address:
    """Represents a physical address."""
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


struct Transaction:
    """Represents a bank transaction."""
    var transaction_id: String
    var account_id: String
    var transaction_type: String
    var amount: Float64
    var timestamp: String
    var description: String
    
    fn __init__(out self, transaction_id: String, account_id: String,
                transaction_type: String, amount: Float64, timestamp: String):
        self.transaction_id = transaction_id
        self.account_id = account_id
        self.transaction_type = transaction_type
        self.amount = amount
        self.timestamp = timestamp
        self.description = ""


struct Account:
    """Represents a bank account."""
    var account_id: String
    var account_type: String
    var balance: Float64
    var owner_id: String
    var status: String
    var interest_rate: Float64
    var transactions: List[String]  # List of transaction IDs
    
    fn __init__(out self, account_id: String, account_type: String, 
                owner_id: String, balance: Float64):
        self.account_id = account_id
        self.account_type = account_type
        self.balance = balance
        self.owner_id = owner_id
        self.status = ACCOUNT_STATUS_ACTIVE
        self.interest_rate = 0.0
        self.transactions = List[String]()
    
    fn is_active(self) -> Bool:
        return self.status == ACCOUNT_STATUS_ACTIVE


struct User:
    """Represents a bank user."""
    var user_id: String
    var first_name: String
    var last_name: String
    var email: String
    var phone: String
    var address: Address
    var accounts: List[String]  # List of account IDs
    var username: String
    var password: String
    
    fn __init__(out self, user_id: String, first_name: String, last_name: String,
                email: String, username: String):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = ""
        self.address = Address("", "", "", "", "")
        self.accounts = List[String]()
        self.username = username
        self.password = ""
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct BankDB:
    """Database for the bank domain."""
    var users: Dict[String, User]
    var accounts: Dict[String, Account]
    var transactions: Dict[String, Transaction]
    
    fn __init__(out self):
        self.users = Dict[String, User]()
        self.accounts = Dict[String, Account]()
        self.transactions = Dict[String, Transaction]()
    
    fn add_user(inout self, user: User):
        self.users[user.user_id] = user
    
    fn add_account(inout self, account: Account):
        self.accounts[account.account_id] = account
    
    fn add_transaction(inout self, transaction: Transaction):
        self.transactions[transaction.transaction_id] = transaction
    
    fn get_user(self, user_id: String) raises -> User:
        if user_id not in self.users:
            raise Error("User " + user_id + " not found")
        return self.users[user_id]
    
    fn get_account(self, account_id: String) raises -> Account:
        if account_id not in self.accounts:
            raise Error("Account " + account_id + " not found")
        return self.accounts[account_id]

