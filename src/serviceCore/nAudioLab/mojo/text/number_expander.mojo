"""
Number to Text Expansion for TTS
Converts numbers to their word representation

Supports:
- Cardinals: 42 → "forty two"
- Ordinals: 42nd → "forty second"
- Decimals: 3.14 → "three point one four"
- Large numbers: 1,234,567 → "one million two hundred thirty four thousand five hundred sixty seven"
- Negative numbers: -42 → "negative forty two"
- Fractions: 1/2 → "one half"
"""

from collections import List


struct NumberExpander:
    """Number to text conversion engine"""
    
    var ones: List[String]
    var teens: List[String]
    var tens: List[String]
    var scales: List[String]
    var ordinal_ones: List[String]
    var ordinal_tens: List[String]
    
    fn __init__(inout self):
        """Initialize number word dictionaries"""
        # Cardinal ones (0-9)
        self.ones = List[String]()
        self.ones.append("zero")
        self.ones.append("one")
        self.ones.append("two")
        self.ones.append("three")
        self.ones.append("four")
        self.ones.append("five")
        self.ones.append("six")
        self.ones.append("seven")
        self.ones.append("eight")
        self.ones.append("nine")
        
        # Teens (10-19)
        self.teens = List[String]()
        self.teens.append("ten")
        self.teens.append("eleven")
        self.teens.append("twelve")
        self.teens.append("thirteen")
        self.teens.append("fourteen")
        self.teens.append("fifteen")
        self.teens.append("sixteen")
        self.teens.append("seventeen")
        self.teens.append("eighteen")
        self.teens.append("nineteen")
        
        # Tens (20-90)
        self.tens = List[String]()
        self.tens.append("")  # 0
        self.tens.append("")  # 10 (handled by teens)
        self.tens.append("twenty")
        self.tens.append("thirty")
        self.tens.append("forty")
        self.tens.append("fifty")
        self.tens.append("sixty")
        self.tens.append("seventy")
        self.tens.append("eighty")
        self.tens.append("ninety")
        
        # Scale words (thousand, million, etc.)
        self.scales = List[String]()
        self.scales.append("")
        self.scales.append("thousand")
        self.scales.append("million")
        self.scales.append("billion")
        self.scales.append("trillion")
        self.scales.append("quadrillion")
        
        # Ordinal ones
        self.ordinal_ones = List[String]()
        self.ordinal_ones.append("zeroth")
        self.ordinal_ones.append("first")
        self.ordinal_ones.append("second")
        self.ordinal_ones.append("third")
        self.ordinal_ones.append("fourth")
        self.ordinal_ones.append("fifth")
        self.ordinal_ones.append("sixth")
        self.ordinal_ones.append("seventh")
        self.ordinal_ones.append("eighth")
        self.ordinal_ones.append("ninth")
        
        # Ordinal tens
        self.ordinal_tens = List[String]()
        self.ordinal_tens.append("")
        self.ordinal_tens.append("")
        self.ordinal_tens.append("twentieth")
        self.ordinal_tens.append("thirtieth")
        self.ordinal_tens.append("fortieth")
        self.ordinal_tens.append("fiftieth")
        self.ordinal_tens.append("sixtieth")
        self.ordinal_tens.append("seventieth")
        self.ordinal_tens.append("eightieth")
        self.ordinal_tens.append("ninetieth")
    
    fn expand_cardinal(self, num: Int) -> String:
        """
        Convert cardinal number to words
        
        Args:
            num: Integer number to convert
            
        Returns:
            Text representation (e.g., "forty two")
        """
        if num == 0:
            return "zero"
        
        if num < 0:
            return "negative " + self.expand_cardinal(-num)
        
        if num < 10:
            return self.ones[num]
        
        if num < 20:
            return self.teens[num - 10]
        
        if num < 100:
            var tens_digit = num // 10
            var ones_digit = num % 10
            if ones_digit == 0:
                return self.tens[tens_digit]
            else:
                return self.tens[tens_digit] + " " + self.ones[ones_digit]
        
        if num < 1000:
            var hundreds = num // 100
            var remainder = num % 100
            var result = self.ones[hundreds] + " hundred"
            if remainder > 0:
                result += " " + self.expand_cardinal(remainder)
            return result
        
        # Handle thousands, millions, billions, etc.
        return self._expand_large_number(num)
    
    fn _expand_large_number(self, num: Int) -> String:
        """Expand numbers >= 1000"""
        var parts = List[String]()
        var scale_index = 0
        var remaining = num
        
        # Break into groups of 3 digits
        while remaining > 0:
            var group = remaining % 1000
            remaining = remaining // 1000
            
            if group > 0:
                var group_text = self._expand_under_thousand(group)
                if scale_index > 0:
                    group_text += " " + self.scales[scale_index]
                parts.append(group_text)
            
            scale_index += 1
        
        # Reverse and join
        var result = ""
        for i in range(len(parts) - 1, -1, -1):
            if result != "":
                result += " "
            result += parts[i]
        
        return result
    
    fn _expand_under_thousand(self, num: Int) -> String:
        """Helper for numbers 1-999"""
        if num < 100:
            return self.expand_cardinal(num)
        
        var hundreds = num // 100
        var remainder = num % 100
        var result = self.ones[hundreds] + " hundred"
        
        if remainder > 0:
            result += " " + self.expand_cardinal(remainder)
        
        return result
    
    fn expand_ordinal(self, num: Int) -> String:
        """
        Convert ordinal number to words
        
        Args:
            num: Integer number to convert
            
        Returns:
            Ordinal text (e.g., "forty second")
        """
        if num < 0:
            return "negative " + self.expand_ordinal(-num)
        
        if num < 10:
            return self.ordinal_ones[num]
        
        if num < 20:
            # Special cases for 10-19
            if num == 10:
                return "tenth"
            elif num == 11:
                return "eleventh"
            elif num == 12:
                return "twelfth"
            else:
                return self.teens[num - 10] + "th"
        
        if num < 100:
            var tens_digit = num // 10
            var ones_digit = num % 10
            if ones_digit == 0:
                return self.ordinal_tens[tens_digit]
            else:
                return self.tens[tens_digit] + " " + self.ordinal_ones[ones_digit]
        
        # For larger numbers, make the last word ordinal
        var cardinal = self.expand_cardinal(num)
        # This is simplified - full implementation would parse and modify last word
        return cardinal + "th"
    
    fn expand_decimal(self, text: String) -> String:
        """
        Expand decimal numbers
        
        Args:
            text: Decimal as string (e.g., "3.14")
            
        Returns:
            Text representation (e.g., "three point one four")
        """
        var parts = text.split(".")
        
        if len(parts) != 2:
            return text  # Not a valid decimal
        
        var integer_part = int(parts[0])
        var integer_text = self.expand_cardinal(integer_part)
        
        var decimal_text = " point"
        for digit_char in parts[1]:
            var digit = int(digit_char)
            decimal_text += " " + self.ones[digit]
        
        return integer_text + decimal_text
    
    fn expand_fraction(self, numerator: Int, denominator: Int) -> String:
        """
        Expand fractions
        
        Args:
            numerator: Top number
            denominator: Bottom number
            
        Returns:
            Text representation (e.g., "one half", "three fourths")
        """
        # Special cases
        if denominator == 2:
            if numerator == 1:
                return "one half"
            else:
                return self.expand_cardinal(numerator) + " halves"
        
        if denominator == 4:
            if numerator == 1:
                return "one quarter"
            elif numerator == 3:
                return "three quarters"
            else:
                return self.expand_cardinal(numerator) + " quarters"
        
        # General case
        var num_text = self.expand_cardinal(numerator)
        var den_text = self.expand_ordinal(denominator)
        
        if numerator > 1:
            den_text += "s"  # Pluralize
        
        return num_text + " " + den_text


fn expand_number(num: Int) -> String:
    """
    Convenience function for cardinal number expansion
    
    Args:
        num: Integer to convert
        
    Returns:
        Text representation
    """
    var expander = NumberExpander()
    return expander.expand_cardinal(num)


fn expand_ordinal_number(num: Int) -> String:
    """
    Convenience function for ordinal number expansion
    
    Args:
        num: Integer to convert
        
    Returns:
        Ordinal text representation
    """
    var expander = NumberExpander()
    return expander.expand_ordinal(num)


fn test_number_expander():
    """Test the number expander"""
    var expander = NumberExpander()
    
    print("Number Expansion Tests:")
    print("=" * 60)
    
    # Cardinals
    var cardinal_tests = [0, 1, 5, 10, 11, 12, 15, 20, 21, 42, 99, 100, 101, 
                          500, 999, 1000, 1234, 10000, 1000000, 1234567]
    
    print("\nCardinal Numbers:")
    print("-" * 60)
    for num in cardinal_tests:
        var text = expander.expand_cardinal(num)
        print(f"{num:>10} → {text}")
    
    # Ordinals
    var ordinal_tests = [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 42, 100]
    
    print("\nOrdinal Numbers:")
    print("-" * 60)
    for num in ordinal_tests:
        var text = expander.expand_ordinal(num)
        print(f"{num:>10} → {text}")
    
    # Decimals
    print("\nDecimal Numbers:")
    print("-" * 60)
    var decimal_tests = ["3.14", "0.5", "10.25", "100.01"]
    for dec in decimal_tests:
        var text = expander.expand_decimal(dec)
        print(f"{dec:>10} → {text}")
    
    # Fractions
    print("\nFractions:")
    print("-" * 60)
    var fraction_tests = [(1, 2), (1, 4), (3, 4), (2, 3), (5, 8)]
    for frac in fraction_tests:
        var text = expander.expand_fraction(frac[0], frac[1])
        print(f"{frac[0]}/{frac[1]:>8} → {text}")
    
    # Negatives
    print("\nNegative Numbers:")
    print("-" * 60)
    var negative_tests = [-1, -42, -100, -1000]
    for num in negative_tests:
        var text = expander.expand_cardinal(num)
        print(f"{num:>10} → {text}")


fn main():
    """Main entry point"""
    test_number_expander()
