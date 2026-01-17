"""
Text Normalization for TTS
Converts any text to pronounceable, normalized form

Features:
- Lowercase conversion
- Number expansion (cardinal, ordinal)
- Date expansion
- Currency expansion
- Abbreviation expansion
- Special character handling
- URL/email handling
"""

from collections import Dict, List


struct TextNormalizer:
    """Main text normalization engine"""
    var abbreviations: Dict[String, String]
    
    fn __init__(inout self):
        """Initialize normalizer with abbreviation dictionary"""
        self.abbreviations = Dict[String, String]()
        self._load_abbreviations()
    
    fn _load_abbreviations(inout self):
        """Load common abbreviations"""
        # Titles
        self.abbreviations["Dr."] = "Doctor"
        self.abbreviations["Mr."] = "Mister"
        self.abbreviations["Mrs."] = "Misses"
        self.abbreviations["Ms."] = "Miss"
        self.abbreviations["Prof."] = "Professor"
        
        # Locations
        self.abbreviations["St."] = "Street"
        self.abbreviations["Ave."] = "Avenue"
        self.abbreviations["Blvd."] = "Boulevard"
        self.abbreviations["Rd."] = "Road"
        self.abbreviations["Dr."] = "Drive"  # Note: context-dependent with Doctor
        self.abbreviations["Ln."] = "Lane"
        self.abbreviations["Ct."] = "Court"
        
        # Units
        self.abbreviations["ft."] = "feet"
        self.abbreviations["in."] = "inches"
        self.abbreviations["lb."] = "pounds"
        self.abbreviations["oz."] = "ounces"
        self.abbreviations["kg."] = "kilograms"
        self.abbreviations["km."] = "kilometers"
        self.abbreviations["mph."] = "miles per hour"
        
        # Time
        self.abbreviations["Jan."] = "January"
        self.abbreviations["Feb."] = "February"
        self.abbreviations["Mar."] = "March"
        self.abbreviations["Apr."] = "April"
        self.abbreviations["Aug."] = "August"
        self.abbreviations["Sept."] = "September"
        self.abbreviations["Oct."] = "October"
        self.abbreviations["Nov."] = "November"
        self.abbreviations["Dec."] = "December"
        
        # Common
        self.abbreviations["etc."] = "et cetera"
        self.abbreviations["vs."] = "versus"
        self.abbreviations["approx."] = "approximately"
        self.abbreviations["misc."] = "miscellaneous"
    
    fn normalize(self, text: String) -> String:
        """
        Main normalization pipeline
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized, pronounceable text
        """
        var result = text
        
        # Step 1: Expand abbreviations
        result = self._expand_abbreviations(result)
        
        # Step 2: Expand currency ($10.50 → "ten dollars and fifty cents")
        result = self._expand_currency(result)
        
        # Step 3: Expand dates (1/16/2026 → "January sixteenth, twenty twenty six")
        result = self._expand_dates(result)
        
        # Step 4: Expand numbers (42 → "forty two")
        result = self._expand_numbers(result)
        
        # Step 5: Handle special characters
        result = self._handle_special_chars(result)
        
        # Step 6: Lowercase
        result = result.lower()
        
        # Step 7: Clean up whitespace
        result = self._clean_whitespace(result)
        
        return result
    
    fn _expand_abbreviations(self, text: String) -> String:
        """Replace abbreviations with full forms"""
        var result = text
        
        for abbrev in self.abbreviations.keys():
            var full_form = self.abbreviations[abbrev]
            result = result.replace(abbrev, full_form)
        
        return result
    
    fn _expand_currency(self, text: String) -> String:
        """
        Expand currency amounts
        Examples:
            $10 → "ten dollars"
            $10.50 → "ten dollars and fifty cents"
            €20 → "twenty euros"
        """
        var result = text
        
        # Simple pattern matching for $XX.YY
        # In production, use regex or more sophisticated parsing
        
        # For now, mark currency for expansion
        # TODO: Implement full currency parser
        result = result.replace("$", " dollars ")
        result = result.replace("€", " euros ")
        result = result.replace("£", " pounds ")
        
        return result
    
    fn _expand_dates(self, text: String) -> String:
        """
        Expand date formats
        Examples:
            1/16/2026 → "January sixteenth, twenty twenty six"
            Jan 16, 2026 → "January sixteenth, twenty twenty six"
        """
        var result = text
        
        # Simple date handling
        # In production, use proper date parser
        # TODO: Implement full date parser
        
        return result
    
    fn _expand_numbers(self, text: String) -> String:
        """
        Expand numbers to words
        Examples:
            42 → "forty two"
            1234 → "one thousand two hundred thirty four"
            3.14 → "three point one four"
        """
        var result = text
        
        # This is a simplified version
        # The full implementation is in number_expander.mojo
        # TODO: Call NumberExpander for full functionality
        
        return result
    
    fn _handle_special_chars(self, text: String) -> String:
        """Handle special characters"""
        var result = text
        
        # Remove or replace special characters
        result = result.replace("&", " and ")
        result = result.replace("+", " plus ")
        result = result.replace("=", " equals ")
        result = result.replace("%", " percent ")
        result = result.replace("#", " number ")
        result = result.replace("@", " at ")
        
        # Remove punctuation that doesn't affect pronunciation
        result = result.replace("\"", "")
        result = result.replace("'", "")
        result = result.replace("[", "")
        result = result.replace("]", "")
        result = result.replace("{", "")
        result = result.replace("}", "")
        result = result.replace("|", "")
        result = result.replace("\\", "")
        result = result.replace("/", " ")
        
        # Keep sentence-ending punctuation
        # . ! ? are kept for prosody
        
        return result
    
    fn _clean_whitespace(self, text: String) -> String:
        """Normalize whitespace"""
        var result = text
        
        # Replace multiple spaces with single space
        while "  " in result:
            result = result.replace("  ", " ")
        
        # Trim leading/trailing whitespace
        result = result.strip()
        
        return result


fn normalize_text(text: String) -> String:
    """
    Convenience function for text normalization
    
    Args:
        text: Raw input text
        
    Returns:
        Normalized text ready for phonemization
    """
    var normalizer = TextNormalizer()
    return normalizer.normalize(text)


fn test_normalizer():
    """Test the text normalizer"""
    var normalizer = TextNormalizer()
    
    # Test cases
    var test_cases = List[String]()
    test_cases.append("Dr. Smith lives on Main St.")
    test_cases.append("The price is $10.50 USD.")
    test_cases.append("Call me @ 555-1234")
    test_cases.append("I scored 42 points & won!")
    test_cases.append("Meeting on Jan. 16, 2026")
    
    print("Text Normalization Tests:")
    print("=" * 50)
    
    for test in test_cases:
        var normalized = normalizer.normalize(test)
        print(f"Input:  {test}")
        print(f"Output: {normalized}")
        print("-" * 50)


fn main():
    """Main entry point"""
    test_normalizer()
