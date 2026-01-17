"""
Phoneme System for TTS
ARPAbet phoneme representation and features

English has 39 phonemes (consonants + vowels):
- 15 vowel phonemes (including diphthongs)
- 24 consonant phonemes

ARPAbet is a phonetic transcription code designed for American English.
Each phoneme has linguistic features that help guide synthesis.
"""

from collections import Dict, List


struct PhonemeFeatures:
    """Articulatory and acoustic features of a phoneme"""
    var voicing: Bool          # Voiced or unvoiced
    var place: String          # Place of articulation
    var manner: String         # Manner of articulation  
    var vowel: Bool           # Is it a vowel?
    var height: String        # Vowel height (for vowels)
    var backness: String      # Vowel backness (for vowels)
    var roundness: Bool       # Vowel roundness (for vowels)
    
    fn __init__(inout self, 
                voicing: Bool = False,
                place: String = "",
                manner: String = "",
                vowel: Bool = False,
                height: String = "",
                backness: String = "",
                roundness: Bool = False):
        self.voicing = voicing
        self.place = place
        self.manner = manner
        self.vowel = vowel
        self.height = height
        self.backness = backness
        self.roundness = roundness


struct Phoneme:
    """Individual phoneme with symbol and features"""
    var symbol: String         # ARPAbet symbol (e.g., "AH0", "K", "AE1")
    var ipa: String           # IPA representation
    var features: PhonemeFeatures
    var stress: Int           # Stress level (0=unstressed, 1=primary, 2=secondary)
    
    fn __init__(inout self, 
                symbol: String,
                ipa: String = "",
                features: PhonemeFeatures = PhonemeFeatures(),
                stress: Int = 0):
        self.symbol = symbol
        self.ipa = ipa
        self.features = features
        self.stress = stress
    
    fn is_vowel(self) -> Bool:
        """Check if this is a vowel phoneme"""
        return self.features.vowel
    
    fn is_consonant(self) -> Bool:
        """Check if this is a consonant phoneme"""
        return not self.features.vowel
    
    fn is_voiced(self) -> Bool:
        """Check if this phoneme is voiced"""
        return self.features.voicing


struct PhonemeSet:
    """Complete set of English phonemes with features"""
    var phonemes: Dict[String, Phoneme]
    
    fn __init__(inout self):
        """Initialize with all English phonemes"""
        self.phonemes = Dict[String, Phoneme]()
        self._load_vowels()
        self._load_consonants()
    
    fn _load_vowels(inout self):
        """Load vowel phonemes"""
        # Monophthongs
        self.phonemes["IY"] = Phoneme("IY", "i", PhonemeFeatures(
            voicing=True, vowel=True, height="close", backness="front", roundness=False))
        
        self.phonemes["IH"] = Phoneme("IH", "ɪ", PhonemeFeatures(
            voicing=True, vowel=True, height="near-close", backness="front", roundness=False))
        
        self.phonemes["EH"] = Phoneme("EH", "ɛ", PhonemeFeatures(
            voicing=True, vowel=True, height="open-mid", backness="front", roundness=False))
        
        self.phonemes["EY"] = Phoneme("EY", "eɪ", PhonemeFeatures(
            voicing=True, vowel=True, height="close-mid", backness="front", roundness=False))
        
        self.phonemes["AE"] = Phoneme("AE", "æ", PhonemeFeatures(
            voicing=True, vowel=True, height="near-open", backness="front", roundness=False))
        
        self.phonemes["AA"] = Phoneme("AA", "ɑ", PhonemeFeatures(
            voicing=True, vowel=True, height="open", backness="back", roundness=False))
        
        self.phonemes["AO"] = Phoneme("AO", "ɔ", PhonemeFeatures(
            voicing=True, vowel=True, height="open-mid", backness="back", roundness=True))
        
        self.phonemes["OW"] = Phoneme("OW", "oʊ", PhonemeFeatures(
            voicing=True, vowel=True, height="close-mid", backness="back", roundness=True))
        
        self.phonemes["UH"] = Phoneme("UH", "ʊ", PhonemeFeatures(
            voicing=True, vowel=True, height="near-close", backness="back", roundness=True))
        
        self.phonemes["UW"] = Phoneme("UW", "u", PhonemeFeatures(
            voicing=True, vowel=True, height="close", backness="back", roundness=True))
        
        self.phonemes["AH"] = Phoneme("AH", "ʌ", PhonemeFeatures(
            voicing=True, vowel=True, height="open-mid", backness="central", roundness=False))
        
        self.phonemes["ER"] = Phoneme("ER", "ɝ", PhonemeFeatures(
            voicing=True, vowel=True, height="mid", backness="central", roundness=False))
        
        # Diphthongs
        self.phonemes["AW"] = Phoneme("AW", "aʊ", PhonemeFeatures(
            voicing=True, vowel=True, height="open", backness="front", roundness=False))
        
        self.phonemes["AY"] = Phoneme("AY", "aɪ", PhonemeFeatures(
            voicing=True, vowel=True, height="open", backness="front", roundness=False))
        
        self.phonemes["OY"] = Phoneme("OY", "ɔɪ", PhonemeFeatures(
            voicing=True, vowel=True, height="open-mid", backness="back", roundness=True))
    
    fn _load_consonants(inout self):
        """Load consonant phonemes"""
        # Stops
        self.phonemes["P"] = Phoneme("P", "p", PhonemeFeatures(
            voicing=False, place="bilabial", manner="stop", vowel=False))
        
        self.phonemes["B"] = Phoneme("B", "b", PhonemeFeatures(
            voicing=True, place="bilabial", manner="stop", vowel=False))
        
        self.phonemes["T"] = Phoneme("T", "t", PhonemeFeatures(
            voicing=False, place="alveolar", manner="stop", vowel=False))
        
        self.phonemes["D"] = Phoneme("D", "d", PhonemeFeatures(
            voicing=True, place="alveolar", manner="stop", vowel=False))
        
        self.phonemes["K"] = Phoneme("K", "k", PhonemeFeatures(
            voicing=False, place="velar", manner="stop", vowel=False))
        
        self.phonemes["G"] = Phoneme("G", "ɡ", PhonemeFeatures(
            voicing=True, place="velar", manner="stop", vowel=False))
        
        # Fricatives
        self.phonemes["F"] = Phoneme("F", "f", PhonemeFeatures(
            voicing=False, place="labiodental", manner="fricative", vowel=False))
        
        self.phonemes["V"] = Phoneme("V", "v", PhonemeFeatures(
            voicing=True, place="labiodental", manner="fricative", vowel=False))
        
        self.phonemes["TH"] = Phoneme("TH", "θ", PhonemeFeatures(
            voicing=False, place="dental", manner="fricative", vowel=False))
        
        self.phonemes["DH"] = Phoneme("DH", "ð", PhonemeFeatures(
            voicing=True, place="dental", manner="fricative", vowel=False))
        
        self.phonemes["S"] = Phoneme("S", "s", PhonemeFeatures(
            voicing=False, place="alveolar", manner="fricative", vowel=False))
        
        self.phonemes["Z"] = Phoneme("Z", "z", PhonemeFeatures(
            voicing=True, place="alveolar", manner="fricative", vowel=False))
        
        self.phonemes["SH"] = Phoneme("SH", "ʃ", PhonemeFeatures(
            voicing=False, place="postalveolar", manner="fricative", vowel=False))
        
        self.phonemes["ZH"] = Phoneme("ZH", "ʒ", PhonemeFeatures(
            voicing=True, place="postalveolar", manner="fricative", vowel=False))
        
        self.phonemes["HH"] = Phoneme("HH", "h", PhonemeFeatures(
            voicing=False, place="glottal", manner="fricative", vowel=False))
        
        # Affricates
        self.phonemes["CH"] = Phoneme("CH", "tʃ", PhonemeFeatures(
            voicing=False, place="postalveolar", manner="affricate", vowel=False))
        
        self.phonemes["JH"] = Phoneme("JH", "dʒ", PhonemeFeatures(
            voicing=True, place="postalveolar", manner="affricate", vowel=False))
        
        # Nasals
        self.phonemes["M"] = Phoneme("M", "m", PhonemeFeatures(
            voicing=True, place="bilabial", manner="nasal", vowel=False))
        
        self.phonemes["N"] = Phoneme("N", "n", PhonemeFeatures(
            voicing=True, place="alveolar", manner="nasal", vowel=False))
        
        self.phonemes["NG"] = Phoneme("NG", "ŋ", PhonemeFeatures(
            voicing=True, place="velar", manner="nasal", vowel=False))
        
        # Liquids
        self.phonemes["L"] = Phoneme("L", "l", PhonemeFeatures(
            voicing=True, place="alveolar", manner="lateral", vowel=False))
        
        self.phonemes["R"] = Phoneme("R", "ɹ", PhonemeFeatures(
            voicing=True, place="alveolar", manner="approximant", vowel=False))
        
        # Glides
        self.phonemes["W"] = Phoneme("W", "w", PhonemeFeatures(
            voicing=True, place="labio-velar", manner="approximant", vowel=False))
        
        self.phonemes["Y"] = Phoneme("Y", "j", PhonemeFeatures(
            voicing=True, place="palatal", manner="approximant", vowel=False))
    
    fn get_phoneme(self, symbol: String) -> Phoneme:
        """Get phoneme by ARPAbet symbol"""
        # Remove stress markers (0, 1, 2) from vowels
        var base_symbol = symbol
        if symbol[-1].isdigit():
            base_symbol = symbol[:-1]
        
        return self.phonemes[base_symbol]
    
    fn is_vowel(self, symbol: String) -> Bool:
        """Check if symbol represents a vowel"""
        var phoneme = self.get_phoneme(symbol)
        return phoneme.is_vowel()
    
    fn is_consonant(self, symbol: String) -> Bool:
        """Check if symbol represents a consonant"""
        var phoneme = self.get_phoneme(symbol)
        return phoneme.is_consonant()
    
    fn count_phonemes(self) -> Int:
        """Get total number of phonemes"""
        return len(self.phonemes)


fn arpabet_to_ipa(arpabet: String) -> String:
    """
    Convert ARPAbet symbol to IPA
    
    Args:
        arpabet: ARPAbet symbol (e.g., "AH0", "K")
        
    Returns:
        IPA symbol
    """
    var phoneme_set = PhonemeSet()
    var phoneme = phoneme_set.get_phoneme(arpabet)
    return phoneme.ipa


fn get_phoneme_features(arpabet: String) -> PhonemeFeatures:
    """
    Get linguistic features for a phoneme
    
    Args:
        arpabet: ARPAbet symbol
        
    Returns:
        Phoneme features
    """
    var phoneme_set = PhonemeSet()
    var phoneme = phoneme_set.get_phoneme(arpabet)
    return phoneme.features


fn test_phoneme_system():
    """Test phoneme system"""
    print("Phoneme System Tests:")
    print("=" * 60)
    
    var phoneme_set = PhonemeSet()
    
    print(f"\nTotal phonemes: {phoneme_set.count_phonemes()}")
    
    # Test vowels
    print("\nVowel Phonemes:")
    print("-" * 60)
    var vowels = ["IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER", "AY", "AW", "OY"]
    for v in vowels:
        var phoneme = phoneme_set.get_phoneme(v)
        print(f"{v:4} → IPA: {phoneme.ipa:3}  Height: {phoneme.features.height}")
    
    # Test consonants
    print("\nConsonant Phonemes:")
    print("-" * 60)
    var consonants = ["P", "B", "T", "D", "K", "G", "F", "V", "S", "Z", "M", "N", "L", "R"]
    for c in consonants:
        var phoneme = phoneme_set.get_phoneme(c)
        var voiced = "voiced" if phoneme.is_voiced() else "unvoiced"
        print(f"{c:4} → IPA: {phoneme.ipa:3}  {phoneme.features.manner:12} {voiced}")
    
    # Test features
    print("\nPhoneme Feature Tests:")
    print("-" * 60)
    print(f"Is 'AH' a vowel? {phoneme_set.is_vowel('AH')}")
    print(f"Is 'K' a vowel? {phoneme_set.is_vowel('K')}")
    print(f"Is 'M' voiced? {phoneme_set.get_phoneme('M').is_voiced()}")
    print(f"Is 'P' voiced? {phoneme_set.get_phoneme('P').is_voiced()}")


fn main():
    """Main entry point"""
    test_phoneme_system()
