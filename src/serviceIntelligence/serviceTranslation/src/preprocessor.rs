/// Text preprocessing for Arabic-English translation

use crate::TranslationPair;
use unicode_normalization::UnicodeNormalization;

pub struct TextPreprocessor {
    normalize_unicode: bool,
    remove_diacritics: bool,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            normalize_unicode: true,
            remove_diacritics: false,
        }
    }
}

impl TextPreprocessor {
    pub fn new(normalize_unicode: bool, remove_diacritics: bool) -> Self {
        Self {
            normalize_unicode,
            remove_diacritics,
        }
    }

    pub fn preprocess(&self, pair: &mut TranslationPair) {
        if self.normalize_unicode {
            pair.arabic = pair.arabic.nfc().collect();
            pair.english = pair.english.nfc().collect();
        }

        if self.remove_diacritics {
            pair.arabic = self.remove_arabic_diacritics(&pair.arabic);
        }

        // Trim whitespace
        pair.arabic = pair.arabic.trim().to_string();
        pair.english = pair.english.trim().to_string();
    }

    fn remove_arabic_diacritics(&self, text: &str) -> String {
        text.chars()
            .filter(|c| !('\u{064B}'..='\u{065F}').contains(c))
            .collect()
    }
}
