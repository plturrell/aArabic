# translation.mojo
# Translation interface for Mojo SDK
# Integrates with TranslateGemma for high-quality multilingual translation

from collections import Dict, List
from .locale import Language, Locale, LanguageRegistry

# ============================================================================
# Translation Request/Response
# ============================================================================

struct TranslationRequest:
    """Request for translation"""
    var text: String
    var source_lang: String
    var target_lang: String
    var context: String  # Optional context for disambiguation

    fn __init__(out self, text: String, source: String, target: String):
        self.text = text
        self.source_lang = source
        self.target_lang = target
        self.context = ""

    fn __init__(out self, text: String, source: String, target: String, context: String):
        self.text = text
        self.source_lang = source
        self.target_lang = target
        self.context = context

struct TranslationResult:
    """Result of translation"""
    var original: String
    var translated: String
    var source_lang: String
    var target_lang: String
    var confidence: Float32
    var success: Bool
    var error_message: String

    fn __init__(out self):
        self.original = ""
        self.translated = ""
        self.source_lang = ""
        self.target_lang = ""
        self.confidence = 0.0
        self.success = False
        self.error_message = ""

    fn __init__(out self, original: String, translated: String, source: String, target: String, confidence: Float32):
        self.original = original
        self.translated = translated
        self.source_lang = source
        self.target_lang = target
        self.confidence = confidence
        self.success = True
        self.error_message = ""

    @staticmethod
    fn error(message: String) -> TranslationResult:
        var result = TranslationResult()
        result.success = False
        result.error_message = message
        return result

# ============================================================================
# Translation Provider Trait
# ============================================================================

trait TranslationProvider:
    """Interface for translation providers"""
    fn translate(self, request: TranslationRequest) -> TranslationResult
    fn supports_language(self, lang_code: String) -> Bool
    fn get_supported_languages(self) -> List[String]

# ============================================================================
# TranslateGemma Provider (Local GGUF)
# ============================================================================

struct TranslateGemmaProvider:
    """
    Translation provider using TranslateGemma-27B-IT model.
    Supports 55 languages with state-of-the-art quality.
    Uses local GGUF inference via HTTP endpoint.
    """
    var endpoint: String
    var model_name: String
    var temperature: Float32
    var max_tokens: Int
    var supported_languages: List[String]

    fn __init__(out self, endpoint: String = "http://localhost:11435/v1/chat/completions"):
        self.endpoint = endpoint
        self.model_name = "translategemma-27b-it"
        self.temperature = 0.1  # Low for accurate translation
        self.max_tokens = 2048
        self._init_languages()

    fn _init_languages(inout self):
        """Initialize supported language list"""
        self.supported_languages = List[String]()
        # 55 languages supported by TranslateGemma
        var langs = [
            "ar", "bn", "cs", "da", "de", "el", "en", "es", "fa", "fi",
            "fil", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "ko",
            "lt", "lv", "mr", "nl", "no", "pl", "pt", "ro", "ru", "sk",
            "sl", "sv", "sw", "ta", "te", "th", "tr", "uk", "ur", "vi",
            "zh", "bg", "ca", "et", "gl", "gu", "kn", "ml", "ms", "my",
            "ne", "pa", "si", "sq", "sr"
        ]
        for i in range(len(langs)):
            self.supported_languages.append(langs[i])

    fn _get_language_name(self, code: String) -> String:
        """Convert language code to full name"""
        var names = Dict[String, String]()
        names["ar"] = "Arabic"
        names["bn"] = "Bengali"
        names["cs"] = "Czech"
        names["da"] = "Danish"
        names["de"] = "German"
        names["el"] = "Greek"
        names["en"] = "English"
        names["es"] = "Spanish"
        names["fa"] = "Persian"
        names["fi"] = "Finnish"
        names["fil"] = "Filipino"
        names["fr"] = "French"
        names["he"] = "Hebrew"
        names["hi"] = "Hindi"
        names["hr"] = "Croatian"
        names["hu"] = "Hungarian"
        names["id"] = "Indonesian"
        names["it"] = "Italian"
        names["ja"] = "Japanese"
        names["ko"] = "Korean"
        names["lt"] = "Lithuanian"
        names["lv"] = "Latvian"
        names["mr"] = "Marathi"
        names["nl"] = "Dutch"
        names["no"] = "Norwegian"
        names["pl"] = "Polish"
        names["pt"] = "Portuguese"
        names["ro"] = "Romanian"
        names["ru"] = "Russian"
        names["sk"] = "Slovak"
        names["sl"] = "Slovenian"
        names["sv"] = "Swedish"
        names["sw"] = "Swahili"
        names["ta"] = "Tamil"
        names["te"] = "Telugu"
        names["th"] = "Thai"
        names["tr"] = "Turkish"
        names["uk"] = "Ukrainian"
        names["ur"] = "Urdu"
        names["vi"] = "Vietnamese"
        names["zh"] = "Chinese"

        if code in names:
            return names[code]
        return code

    fn _build_prompt(self, text: String, source: String, target: String) -> String:
        """Build TranslateGemma prompt format"""
        let source_name = self._get_language_name(source)
        let target_name = self._get_language_name(target)
        return "<source_lang>" + source_name + "</source_lang><target_lang>" + target_name + "</target_lang>" + text

    fn supports_language(self, lang_code: String) -> Bool:
        """Check if language is supported"""
        for i in range(len(self.supported_languages)):
            if self.supported_languages[i] == lang_code:
                return True
        return False

    fn get_supported_languages(self) -> List[String]:
        """Get list of supported language codes"""
        return self.supported_languages

    fn translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate text using TranslateGemma"""
        # Validate languages
        if not self.supports_language(request.source_lang):
            return TranslationResult.error("Source language not supported: " + request.source_lang)
        if not self.supports_language(request.target_lang):
            return TranslationResult.error("Target language not supported: " + request.target_lang)

        # Build prompt
        let prompt = self._build_prompt(request.text, request.source_lang, request.target_lang)

        # Call local inference (using native HTTP client)
        let translated = self._call_inference(prompt)

        if len(translated) > 0:
            return TranslationResult(
                request.text,
                translated,
                request.source_lang,
                request.target_lang,
                0.95  # High confidence for TranslateGemma
            )
        else:
            return TranslationResult.error("Translation failed")

    fn _call_inference(self, prompt: String) -> String:
        """Call local inference endpoint"""
        # This would use native HTTP client from stdlib
        # For now, return empty to indicate need for implementation
        # In production, this uses the llm_http_service
        return ""

    fn translate_batch(self, requests: List[TranslationRequest]) -> List[TranslationResult]:
        """Translate multiple texts"""
        var results = List[TranslationResult]()
        for i in range(len(requests)):
            results.append(self.translate(requests[i]))
        return results

# ============================================================================
# SDK Translation Manager
# ============================================================================

struct TranslationManager:
    """
    Central translation manager for the SDK.
    Manages providers and caching.
    """
    var provider: TranslateGemmaProvider
    var cache: Dict[String, String]
    var cache_enabled: Bool
    var max_cache_size: Int

    fn __init__(out self):
        self.provider = TranslateGemmaProvider()
        self.cache = Dict[String, String]()
        self.cache_enabled = True
        self.max_cache_size = 10000

    fn __init__(out self, endpoint: String):
        self.provider = TranslateGemmaProvider(endpoint)
        self.cache = Dict[String, String]()
        self.cache_enabled = True
        self.max_cache_size = 10000

    fn _cache_key(self, text: String, source: String, target: String) -> String:
        """Generate cache key"""
        return source + "|" + target + "|" + text

    fn translate(inout self, text: String, source_lang: String, target_lang: String) -> TranslationResult:
        """Translate text with caching"""
        # Check cache
        if self.cache_enabled:
            let key = self._cache_key(text, source_lang, target_lang)
            if key in self.cache:
                return TranslationResult(
                    text,
                    self.cache[key],
                    source_lang,
                    target_lang,
                    1.0  # Cache hit
                )

        # Translate
        let request = TranslationRequest(text, source_lang, target_lang)
        let result = self.provider.translate(request)

        # Cache successful translation
        if result.success and self.cache_enabled:
            let key = self._cache_key(text, source_lang, target_lang)
            if len(self.cache) < self.max_cache_size:
                self.cache[key] = result.translated

        return result

    fn translate_to(inout self, text: String, target_lang: String, source_lang: String = "auto") -> String:
        """Convenience method for translation"""
        var src = source_lang
        if src == "auto":
            src = self._detect_language(text)

        let result = self.translate(text, src, target_lang)
        if result.success:
            return result.translated
        return text  # Return original on failure

    fn _detect_language(self, text: String) -> String:
        """Simple language detection based on character ranges"""
        if len(text) == 0:
            return "en"

        # Check for Arabic characters
        for i in range(len(text)):
            let c = ord(text[i])
            if c >= 0x0600 and c <= 0x06FF:
                return "ar"
            if c >= 0x4E00 and c <= 0x9FFF:
                return "zh"
            if c >= 0x3040 and c <= 0x309F:
                return "ja"
            if c >= 0xAC00 and c <= 0xD7AF:
                return "ko"
            if c >= 0x0400 and c <= 0x04FF:
                return "ru"
            if c >= 0x0590 and c <= 0x05FF:
                return "he"

        return "en"  # Default to English

    fn clear_cache(inout self):
        """Clear translation cache"""
        self.cache = Dict[String, String]()

    fn get_cache_stats(self) -> (Int, Int):
        """Get cache statistics (size, max_size)"""
        return (len(self.cache), self.max_cache_size)

    fn supports_language(self, lang_code: String) -> Bool:
        """Check if language is supported"""
        return self.provider.supports_language(lang_code)

    fn get_supported_languages(self) -> List[String]:
        """Get list of supported languages"""
        return self.provider.get_supported_languages()

# ============================================================================
# Global Translation Instance
# ============================================================================

var _global_manager: TranslationManager = TranslationManager()

fn get_translator() -> TranslationManager:
    """Get the global translation manager"""
    return _global_manager

fn translate(text: String, source: String, target: String) -> String:
    """Translate text (global function)"""
    let result = _global_manager.translate(text, source, target)
    if result.success:
        return result.translated
    return text

fn t(text: String, target: String) -> String:
    """Shorthand for translation with auto-detect source"""
    return _global_manager.translate_to(text, target)

# ============================================================================
# SDK Integration Helpers
# ============================================================================

struct LocalizedString:
    """String wrapper with automatic translation support"""
    var original: String
    var translations: Dict[String, String]

    fn __init__(out self, text: String):
        self.original = text
        self.translations = Dict[String, String]()

    fn get(self, lang: String = "en") -> String:
        """Get translation for language"""
        if lang in self.translations:
            return self.translations[lang]
        if lang == "en":
            return self.original
        # Translate on demand
        return translate(self.original, "en", lang)

    fn preload(inout self, languages: List[String]):
        """Preload translations for specified languages"""
        for i in range(len(languages)):
            let lang = languages[i]
            if lang not in self.translations and lang != "en":
                self.translations[lang] = translate(self.original, "en", lang)

    fn __str__(self) -> String:
        return self.original

# Helper function to create localized strings
fn L(text: String) -> LocalizedString:
    """Create a localizable string"""
    return LocalizedString(text)
