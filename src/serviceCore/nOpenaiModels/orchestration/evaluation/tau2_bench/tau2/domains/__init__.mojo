# Tau2 Domains - Pure Mojo Implementation
# This module provides domain definitions for the tau2 evaluation framework

from collections import Dict, List

# Domain registry for accessing domain environments and tools
struct DomainRegistry:
    """Registry for tau2 evaluation domains."""
    
    var domains: List[String]
    
    fn __init__(out self):
        """Initialize the domain registry with available domains."""
        self.domains = List[String]()
        self.domains.append("mock")
        self.domains.append("airline")
        self.domains.append("bank")
        self.domains.append("basketball")
        self.domains.append("ecommerce")
        self.domains.append("medicine")
        self.domains.append("movie")
        self.domains.append("railway")
        self.domains.append("restaurant")
        self.domains.append("retail")
        self.domains.append("school")
        self.domains.append("telecom")
        self.domains.append("travel")
        self.domains.append("weather")
    
    fn list_domains(self) -> List[String]:
        """Return a list of all available domain names."""
        return self.domains
    
    fn has_domain(self, name: String) -> Bool:
        """Check if a domain exists by name."""
        for i in range(len(self.domains)):
            if self.domains[i] == name:
                return True
        return False
    
    fn domain_count(self) -> Int:
        """Return the number of registered domains."""
        return len(self.domains)


fn get_registry() -> DomainRegistry:
    """Get the domain registry instance."""
    return DomainRegistry()

