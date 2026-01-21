# tau2/registry.mojo
# Migrated from tau2/registry.py
# Domain registry for TAU2-Bench

from collections import Dict, List

struct DomainInfo:
    """Information about a registered domain."""
    var name: String
    var description: String
    var available: Bool
    
    fn __init__(inout self, name: String, description: String = "", available: Bool = True):
        self.name = name
        self.description = description
        self.available = available

struct DomainRegistry:
    """Registry of available domains for TAU2-Bench evaluation."""
    var domains: Dict[String, DomainInfo]
    
    fn __init__(inout self):
        self.domains = Dict[String, DomainInfo]()
        self._register_default_domains()
    
    fn _register_default_domains(inout self):
        """Register the default domains."""
        # General Topics
        self.register_domain("data", "Data management and analytics")
        self.register_domain("process", "Business process automation and optimization")
        self.register_domain("insight", "Business intelligence and insights generation")
        
        # Business Topics - Financial Services
        self.register_domain("esg_sustainable_finance", "ESG and Sustainable Finance")
        self.register_domain("financial_performance", "Financial Business Performance Analysis")
        self.register_domain("treasury_liquidity", "Treasury, Liquidity and Balance Sheet Management")
        self.register_domain("accounts_payable", "Accounts Payable")
        self.register_domain("regulatory_reporting", "Local Regulatory Reporting")
        self.register_domain("financial_disclosure", "Financial Disclosure and Controllership")
        
        # Additional domains can be registered dynamically
    
    fn register_domain(inout self, name: String, description: String = ""):
        """Register a new domain."""
        var info = DomainInfo(name, description)
        self.domains[name] = info
    
    fn get_domain(self, name: String) -> DomainInfo:
        """Get domain information."""
        if name in self.domains:
            return self.domains[name]
        return DomainInfo("unknown", "Domain not found", False)
    
    fn list_domains(self) -> List[String]:
        """List all registered domains."""
        var result = List[String]()
        # TODO: Implement when Dict iteration is available
        return result

# Global registry instance
var GLOBAL_REGISTRY = DomainRegistry()

fn get_registry() -> DomainRegistry:
    """Get the global domain registry."""
    return GLOBAL_REGISTRY
