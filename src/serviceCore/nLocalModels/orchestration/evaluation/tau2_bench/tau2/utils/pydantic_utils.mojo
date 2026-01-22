# pydantic_utils.mojo
# Migrated from pydantic_utils.py
# JSON validation and schema utilities (Pydantic replacement)

from collections import Dict, List

struct ValidationError:
    """Validation error information"""
    var field: String
    var message: String
    var value: String
    
    fn __init__(out self):
        self.field = ""
        self.message = ""
        self.value = ""
    
    fn __init__(out self, field: String, message: String, value: String):
        self.field = field
        self.message = message
        self.value = value
    
    fn to_string(self) -> String:
        """Convert to string representation"""
        return "ValidationError in field '" + self.field + "': " + self.message + " (value: " + self.value + ")"

struct ValidationResult:
    """Result of validation operation"""
    var is_valid: Bool
    var errors: List[ValidationError]
    
    fn __init__(out self):
        self.is_valid = True
        self.errors = List[ValidationError]()
    
    fn add_error(mut self, field: String, message: String, value: String = ""):
        """Add a validation error"""
        self.is_valid = False
        self.errors.append(ValidationError(field, message, value))
    
    fn get_error_count(self) -> Int:
        """Get number of errors"""
        return len(self.errors)
    
    fn get_errors_string(self) -> String:
        """Get all errors as string"""
        var result = ""
        for i in range(len(self.errors)):
            result = result + self.errors[i].to_string() + "\n"
        return result

struct FieldValidator:
    """Validator for a single field"""
    var field_name: String
    var field_type: String  # "string", "int", "float", "bool", "list", "dict"
    var required: Bool
    var min_length: Int
    var max_length: Int
    var min_value: Float32
    var max_value: Float32
    var allowed_values: List[String]
    
    fn __init__(out self):
        self.field_name = ""
        self.field_type = "string"
        self.required = False
        self.min_length = 0
        self.max_length = -1  # -1 means no limit
        self.min_value = -999999.0
        self.max_value = 999999.0
        self.allowed_values = List[String]()
    
    fn __init__(out self, field_name: String, field_type: String, required: Bool):
        self.field_name = field_name
        self.field_type = field_type
        self.required = required
        self.min_length = 0
        self.max_length = -1
        self.min_value = -999999.0
        self.max_value = 999999.0
        self.allowed_values = List[String]()
    
    fn validate_string(self, value: String) -> ValidationResult:
        """Validate string value"""
        var result = ValidationResult()
        
        if self.min_length > 0 and len(value) < self.min_length:
            result.add_error(self.field_name, "String too short (min: " + str(self.min_length) + ")", value)
        
        if self.max_length > 0 and len(value) > self.max_length:
            result.add_error(self.field_name, "String too long (max: " + str(self.max_length) + ")", value)
        
        if len(self.allowed_values) > 0:
            var found = False
            for i in range(len(self.allowed_values)):
                if value == self.allowed_values[i]:
                    found = True
                    break
            if not found:
                result.add_error(self.field_name, "Value not in allowed list", value)
        
        return result
    
    fn set_length_range(mut self, min_len: Int, max_len: Int):
        """Set length constraints"""
        self.min_length = min_len
        self.max_length = max_len
    
    fn set_value_range(mut self, min_val: Float32, max_val: Float32):
        """Set value range constraints"""
        self.min_value = min_val
        self.max_value = max_val
    
    fn add_allowed_value(mut self, value: String):
        """Add an allowed value"""
        self.allowed_values.append(value)

struct SchemaValidator:
    """Validator for a complete schema"""
    var schema_name: String
    var fields: Dict[String, FieldValidator]
    var strict_mode: Bool  # Reject unknown fields
    
    fn __init__(out self):
        self.schema_name = ""
        self.fields = Dict[String, FieldValidator]()
        self.strict_mode = False
    
    fn __init__(out self, schema_name: String, strict_mode: Bool = False):
        self.schema_name = schema_name
        self.fields = Dict[String, FieldValidator]()
        self.strict_mode = strict_mode
    
    fn add_field(mut self, validator: FieldValidator):
        """Add a field validator"""
        self.fields[validator.field_name] = validator
    
    fn validate_dict(self, data: Dict[String, String]) -> ValidationResult:
        """Validate a dictionary against the schema"""
        var result = ValidationResult()
        
        # Check required fields
        for field_name in self.fields:
            let validator = self.fields[field_name]
            if validator.required and field_name not in data:
                result.add_error(field_name, "Required field missing", "")
        
        # Validate present fields
        for field_name in data:
            if field_name not in self.fields:
                if self.strict_mode:
                    result.add_error(field_name, "Unknown field in strict mode", data[field_name])
                continue
            
            let validator = self.fields[field_name]
            let value = data[field_name]
            
            # Validate based on type
            if validator.field_type == "string":
                let field_result = validator.validate_string(value)
                if not field_result.is_valid:
                    for i in range(len(field_result.errors)):
                        result.errors.append(field_result.errors[i])
                    result.is_valid = False
        
        return result
    
    fn get_field_count(self) -> Int:
        """Get number of fields in schema"""
        return len(self.fields)

fn validate_json_string(json_str: String) -> Bool:
    """
    Basic JSON string validation
    
    Args:
        json_str: JSON string to validate
        
    Returns:
        True if valid JSON structure
    """
    # Simple validation - check for balanced braces
    var brace_count = 0
    var bracket_count = 0
    
    for i in range(len(json_str)):
        let char = json_str[i]
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
        elif char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
        
        if brace_count < 0 or bracket_count < 0:
            return False
    
    return brace_count == 0 and bracket_count == 0

fn validate_email(email: String) -> Bool:
    """
    Basic email validation
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid email format
    """
    # Simple check for @ and .
    let has_at = "@" in email
    let has_dot = "." in email
    let at_pos = email.find("@")
    let dot_pos = email.rfind(".")
    
    return has_at and has_dot and at_pos > 0 and dot_pos > at_pos

fn validate_url(url: String) -> Bool:
    """
    Basic URL validation
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL format
    """
    let has_protocol = url.startswith("http://") or url.startswith("https://")
    let has_domain = "." in url
    
    return has_protocol and has_domain and len(url) > 10

fn create_string_field(name: String, required: Bool = False, 
                       min_length: Int = 0, max_length: Int = -1) -> FieldValidator:
    """
    Create a string field validator
    
    Args:
        name: Field name
        required: Whether field is required
        min_length: Minimum length
        max_length: Maximum length
        
    Returns:
        Configured FieldValidator
    """
    var validator = FieldValidator(name, "string", required)
    validator.set_length_range(min_length, max_length)
    return validator

fn create_enum_field(name: String, allowed_values: List[String], 
                     required: Bool = False) -> FieldValidator:
    """
    Create an enum field validator
    
    Args:
        name: Field name
        allowed_values: List of allowed values
        required: Whether field is required
        
    Returns:
        Configured FieldValidator
    """
    var validator = FieldValidator(name, "string", required)
    for i in range(len(allowed_values)):
        validator.add_allowed_value(allowed_values[i])
    return validator

fn create_number_field(name: String, required: Bool = False,
                       min_value: Float32 = -999999.0, 
                       max_value: Float32 = 999999.0) -> FieldValidator:
    """
    Create a numeric field validator
    
    Args:
        name: Field name
        required: Whether field is required
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Configured FieldValidator
    """
    var validator = FieldValidator(name, "float", required)
    validator.set_value_range(min_value, max_value)
    return validator
