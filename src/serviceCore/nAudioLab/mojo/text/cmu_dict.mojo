"""
CMU Pronouncing Dictionary Loader
Loads and provides access to the CMU Pronouncing Dictionary

The CMU Pronouncing Dictionary contains over 134,000 words with their
phonetic transcriptions in ARPAbet format.

Format: WORD  P1 P2 P3 ...
Example: HELLO  HH AH0 L OW1
"""

from collections import Dict, List


struct CMUDictEntry:
    """Single pronunciation entry"""
    var word: String
    var phonemes: List[String]  # ARPAbet phonemes
    var variant: Int            # Variant number (for multiple pronunciations)
    
    fn __init__(inout self, word: String, phonemes: List[String], variant: Int = 0):
        self.word = word
        self.phonemes = phonemes
        self.variant = variant
    
    fn phoneme_count(self) -> Int:
        """Get number of phonemes"""
        return len(self.phonemes)
    
    fn to_string(self) -> String:
        """Convert to string representation"""
        return self.word + " → " + " ".join(self.phonemes)


struct CMUDict:
    """CMU Pronouncing Dictionary"""
    var entries: Dict[String, List[CMUDictEntry]]
    var loaded: Bool
    
    fn __init__(inout self):
        """Initialize empty dictionary"""
        self.entries = Dict[String, List[CMUDictEntry]]()
        self.loaded = False
    
    fn load(inout self, path: String) raises:
        """
        Load CMU dictionary from file
        
        Args:
            path: Path to cmudict file
            
        Format:
            WORD  P1 P2 P3
            WORD(2)  P1 P2 P3  # Alternative pronunciation
        """
        # In production, this would read from file
        # For now, we'll provide a stub that can be filled in
        # when file I/O is available
        
        # Example entries for testing
        self._load_sample_entries()
        self.loaded = True
    
    fn _load_sample_entries(inout self):
        """Load sample entries for testing"""
        # Common words
        self.add_entry("HELLO", ["HH", "AH0", "L", "OW1"])
        self.add_entry("WORLD", ["W", "ER1", "L", "D"])
        self.add_entry("THE", ["DH", "AH0"])
        self.add_entry("THE", ["DH", "IY1"], variant=1)  # Alternative pronunciation
        self.add_entry("A", ["AH0"])
        self.add_entry("A", ["EY1"], variant=1)
        self.add_entry("IS", ["IH1", "Z"])
        self.add_entry("OF", ["AH1", "V"])
        self.add_entry("TO", ["T", "UW1"])
        self.add_entry("TO", ["T", "AH0"], variant=1)
        self.add_entry("AND", ["AE1", "N", "D"])
        self.add_entry("IN", ["IH1", "N"])
        self.add_entry("FOR", ["F", "AO1", "R"])
        self.add_entry("ON", ["AA1", "N"])
        self.add_entry("WITH", ["W", "IH1", "DH"])
        self.add_entry("AS", ["AE1", "Z"])
        self.add_entry("AT", ["AE1", "T"])
        self.add_entry("BY", ["B", "AY1"])
        self.add_entry("FROM", ["F", "R", "AH1", "M"])
        self.add_entry("THIS", ["DH", "IH1", "S"])
        self.add_entry("THAT", ["DH", "AE1", "T"])
        
        # Numbers
        self.add_entry("ONE", ["W", "AH1", "N"])
        self.add_entry("TWO", ["T", "UW1"])
        self.add_entry("THREE", ["TH", "R", "IY1"])
        self.add_entry("FOUR", ["F", "AO1", "R"])
        self.add_entry("FIVE", ["F", "AY1", "V"])
        self.add_entry("SIX", ["S", "IH1", "K", "S"])
        self.add_entry("SEVEN", ["S", "EH1", "V", "AH0", "N"])
        self.add_entry("EIGHT", ["EY1", "T"])
        self.add_entry("NINE", ["N", "AY1", "N"])
        self.add_entry("TEN", ["T", "EH1", "N"])
        
        # Common verbs
        self.add_entry("MAKE", ["M", "EY1", "K"])
        self.add_entry("TAKE", ["T", "EY1", "K"])
        self.add_entry("GIVE", ["G", "IH1", "V"])
        self.add_entry("WORK", ["W", "ER1", "K"])
        self.add_entry("CALL", ["K", "AO1", "L"])
        self.add_entry("FIND", ["F", "AY1", "N", "D"])
        self.add_entry("THINK", ["TH", "IH1", "NG", "K"])
        self.add_entry("KNOW", ["N", "OW1"])
        self.add_entry("WANT", ["W", "AA1", "N", "T"])
        self.add_entry("NEED", ["N", "IY1", "D"])
        
        # Common adjectives
        self.add_entry("GOOD", ["G", "UH1", "D"])
        self.add_entry("GREAT", ["G", "R", "EY1", "T"])
        self.add_entry("SMALL", ["S", "M", "AO1", "L"])
        self.add_entry("LARGE", ["L", "AA1", "R", "JH"])
        self.add_entry("BIG", ["B", "IH1", "G"])
        self.add_entry("NEW", ["N", "UW1"])
        self.add_entry("OLD", ["OW1", "L", "D"])
        self.add_entry("LONG", ["L", "AO1", "NG"])
        self.add_entry("SHORT", ["SH", "AO1", "R", "T"])
        
        # Days of week
        self.add_entry("MONDAY", ["M", "AH1", "N", "D", "EY2"])
        self.add_entry("TUESDAY", ["T", "UW1", "Z", "D", "EY2"])
        self.add_entry("WEDNESDAY", ["W", "EH1", "N", "Z", "D", "EY2"])
        self.add_entry("THURSDAY", ["TH", "ER1", "Z", "D", "EY2"])
        self.add_entry("FRIDAY", ["F", "R", "AY1", "D", "EY2"])
        self.add_entry("SATURDAY", ["S", "AE1", "T", "ER0", "D", "EY2"])
        self.add_entry("SUNDAY", ["S", "AH1", "N", "D", "EY2"])
    
    fn add_entry(inout self, word: String, phonemes: List[String], variant: Int = 0):
        """Add a dictionary entry"""
        var entry = CMUDictEntry(word, phonemes, variant)
        
        if word not in self.entries:
            self.entries[word] = List[CMUDictEntry]()
        
        self.entries[word].append(entry)
    
    fn lookup(self, word: String) -> List[CMUDictEntry]:
        """
        Look up a word in the dictionary
        
        Args:
            word: Word to look up (case-insensitive)
            
        Returns:
            List of pronunciations (empty if not found)
        """
        var upper_word = word.upper()
        
        if upper_word in self.entries:
            return self.entries[upper_word]
        else:
            return List[CMUDictEntry]()
    
    fn has_word(self, word: String) -> Bool:
        """Check if dictionary contains a word"""
        var upper_word = word.upper()
        return upper_word in self.entries
    
    fn get_primary_pronunciation(self, word: String) -> List[String]:
        """
        Get the primary (first) pronunciation for a word
        
        Args:
            word: Word to look up
            
        Returns:
            List of phonemes (empty if not found)
        """
        var entries = self.lookup(word)
        
        if len(entries) > 0:
            return entries[0].phonemes
        else:
            return List[String]()
    
    fn count_entries(self) -> Int:
        """Get total number of word entries"""
        return len(self.entries)
    
    fn count_variants(self) -> Int:
        """Get total number of pronunciation variants"""
        var total = 0
        for word in self.entries.keys():
            total += len(self.entries[word])
        return total


fn text_to_phonemes(text: String, cmu_dict: CMUDict) -> List[String]:
    """
    Convert text to phoneme sequence
    
    Args:
        text: Input text (normalized, space-separated words)
        cmu_dict: Loaded CMU dictionary
        
    Returns:
        List of phonemes
    """
    var phonemes = List[String]()
    var words = text.split()
    
    for word in words:
        var pronunciation = cmu_dict.get_primary_pronunciation(word)
        
        if len(pronunciation) > 0:
            # Add phonemes from dictionary
            for phoneme in pronunciation:
                phonemes.append(phoneme)
        else:
            # Word not in dictionary - would need G2P here
            # For now, add a placeholder
            phonemes.append("UNK")
    
    return phonemes


fn test_cmu_dict():
    """Test CMU dictionary"""
    print("CMU Dictionary Tests:")
    print("=" * 60)
    
    var cmu_dict = CMUDict()
    
    # Load (sample) dictionary
    try:
        cmu_dict.load("data/phonemes/cmudict.txt")
        print(f"✓ Dictionary loaded")
        print(f"  Total entries: {cmu_dict.count_entries()}")
        print(f"  Total variants: {cmu_dict.count_variants()}")
    except:
        print("✗ Failed to load dictionary")
        return
    
    # Test lookups
    print("\nWord Lookups:")
    print("-" * 60)
    var test_words = ["hello", "world", "the", "one", "two", "three"]
    
    for word in test_words:
        var entries = cmu_dict.lookup(word)
        if len(entries) > 0:
            print(f"{word:10} → {' '.join(entries[0].phonemes)}")
            if len(entries) > 1:
                print(f"{'':10}   (+ {len(entries) - 1} variant(s))")
        else:
            print(f"{word:10} → NOT FOUND")
    
    # Test sentence conversion
    print("\nSentence to Phonemes:")
    print("-" * 60)
    var sentences = [
        "hello world",
        "the quick brown fox",
        "one two three"
    ]
    
    for sentence in sentences:
        var phonemes = text_to_phonemes(sentence, cmu_dict)
        print(f"Text:     {sentence}")
        print(f"Phonemes: {' '.join(phonemes)}")
        print()
    
    # Test multiple pronunciations
    print("\nMultiple Pronunciations:")
    print("-" * 60)
    var entries = cmu_dict.lookup("the")
    for i, entry in enumerate(entries):
        print(f"Variant {i + 1}: {' '.join(entry.phonemes)}")


fn main():
    """Main entry point"""
    test_cmu_dict()
