# Telecom Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the telecom domain

from collections import Dict, List

# Plan type constants
alias PLAN_TYPE_PREPAID = "prepaid"
alias PLAN_TYPE_POSTPAID = "postpaid"

# Account status constants
alias ACCOUNT_STATUS_ACTIVE = "active"
alias ACCOUNT_STATUS_SUSPENDED = "suspended"
alias ACCOUNT_STATUS_CANCELLED = "cancelled"


struct Plan:
    """Represents a telecom plan."""
    var plan_id: String
    var name: String
    var plan_type: String
    var monthly_cost: Float64
    var data_limit_gb: Int
    var minutes: Int
    var texts: Int
    var features: List[String]
    
    fn __init__(out self, plan_id: String, name: String, plan_type: String,
                monthly_cost: Float64):
        self.plan_id = plan_id
        self.name = name
        self.plan_type = plan_type
        self.monthly_cost = monthly_cost
        self.data_limit_gb = 0
        self.minutes = 0
        self.texts = 0
        self.features = List[String]()


struct Device:
    """Represents a telecom device."""
    var device_id: String
    var name: String
    var brand: String
    var model: String
    var price: Float64
    var monthly_payment: Float64
    var in_stock: Bool
    
    fn __init__(out self, device_id: String, name: String, brand: String,
                price: Float64):
        self.device_id = device_id
        self.name = name
        self.brand = brand
        self.model = ""
        self.price = price
        self.monthly_payment = 0.0
        self.in_stock = True


struct Bill:
    """Represents a telecom bill."""
    var bill_id: String
    var account_id: String
    var amount: Float64
    var due_date: String
    var paid: Bool
    var billing_period: String
    
    fn __init__(out self, bill_id: String, account_id: String, amount: Float64,
                due_date: String):
        self.bill_id = bill_id
        self.account_id = account_id
        self.amount = amount
        self.due_date = due_date
        self.paid = False
        self.billing_period = ""


struct Account:
    """Represents a telecom account."""
    var account_id: String
    var user_id: String
    var plan_id: String
    var phone_number: String
    var status: String
    var balance: Float64
    var data_used_gb: Float64
    var minutes_used: Int
    var texts_used: Int
    var bills: List[String]
    
    fn __init__(out self, account_id: String, user_id: String, plan_id: String,
                phone_number: String):
        self.account_id = account_id
        self.user_id = user_id
        self.plan_id = plan_id
        self.phone_number = phone_number
        self.status = ACCOUNT_STATUS_ACTIVE
        self.balance = 0.0
        self.data_used_gb = 0.0
        self.minutes_used = 0
        self.texts_used = 0
        self.bills = List[String]()


struct User:
    """Represents a telecom domain user."""
    var user_id: String
    var first_name: String
    var last_name: String
    var email: String
    var phone: String
    var address: String
    var accounts: List[String]
    
    fn __init__(out self, user_id: String, first_name: String, last_name: String,
                email: String):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.phone = ""
        self.address = ""
        self.accounts = List[String]()
    
    fn full_name(self) -> String:
        return self.first_name + " " + self.last_name


struct TelecomDB:
    """Database for the telecom domain."""
    var plans: Dict[String, Plan]
    var devices: Dict[String, Device]
    var accounts: Dict[String, Account]
    var bills: Dict[String, Bill]
    var users: Dict[String, User]
    
    fn __init__(out self):
        self.plans = Dict[String, Plan]()
        self.devices = Dict[String, Device]()
        self.accounts = Dict[String, Account]()
        self.bills = Dict[String, Bill]()
        self.users = Dict[String, User]()
    
    fn add_plan(inout self, plan: Plan):
        self.plans[plan.plan_id] = plan
    
    fn add_device(inout self, device: Device):
        self.devices[device.device_id] = device
    
    fn add_account(inout self, account: Account):
        self.accounts[account.account_id] = account
    
    fn get_plan(self, plan_id: String) raises -> Plan:
        if plan_id not in self.plans:
            raise Error("Plan " + plan_id + " not found")
        return self.plans[plan_id]
    
    fn get_account(self, account_id: String) raises -> Account:
        if account_id not in self.accounts:
            raise Error("Account " + account_id + " not found")
        return self.accounts[account_id]

