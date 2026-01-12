// Translation Service - Connected via APISIX Gateway
// All requests go through http://localhost/translate

const TRANSLATION_API = 'http://localhost/translate';

/**
 * Translate text between Arabic and English
 * @param {string} text - Text to translate
 * @param {string} direction - Translation direction: 'ar-en' or 'en-ar'
 * @returns {Promise<Object>} Translation result
 */
export async function translateText(text, direction) {
  try {
    // Use the smart endpoint for intelligent routing
    const response = await fetch(`${TRANSLATION_API}/smart`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        source_language: direction === 'ar-en' ? 'ar' : 'en',
        target_language: direction === 'ar-en' ? 'en' : 'ar'
      })
    });

    if (!response.ok) {
      throw new Error(`Translation service returned ${response.status}`);
    }

    const data = await response.json();
    
    return {
      source: text,
      translation: data.translated_text,
      direction,
      // Optional metadata from smart endpoint
      model_used: data.model_used,
      confidence: data.confidence,
      classification: data.classification,
      cached: data.cached
    };
    
  } catch (error) {
    console.error('Translation error:', error);
    
    // Try fallback to basic endpoint
    try {
      const response = await fetch(`${TRANSLATION_API}/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          source_language: direction === 'ar-en' ? 'ar' : 'en',
          target_language: direction === 'ar-en' ? 'en' : 'ar'
        })
      });

      if (!response.ok) {
        throw new Error(`Translation service returned ${response.status}`);
      }

      const data = await response.json();
      
      return {
        source: text,
        translation: data.translated_text,
        direction
      };
    } catch (fallbackError) {
      throw new Error(
        `Translation service unavailable at ${TRANSLATION_API}. ` +
        `Please ensure the platform services are running via the gateway.`
      );
    }
  }
}

/**
 * Generate text embeddings using the embedding service
 * @param {string} text - Text to embed
 * @returns {Promise<Array>} Embedding vector
 */
export async function generateEmbedding(text) {
  try {
    const response = await fetch('http://localhost/embed', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Embedding service error: ${response.status}`);
    }

    const data = await response.json();
    return data.embedding;
  } catch (error) {
    console.warn('Embedding service unavailable:', error.message);
    return null;
  }
}

/**
 * Auto-detect language of input text
 * @param {string} text - Text to analyze
 * @returns {string} Detected language code ('ar' or 'en')
 */
export function detectLanguage(text) {
  // Check for Arabic characters (Unicode range U+0600 to U+06FF)
  const arabicPattern = /[\u0600-\u06FF]/;
  return arabicPattern.test(text) ? 'ar' : 'en';
}

/**
 * Validate text input
 * @param {string} text - Text to validate
 * @returns {Object} Validation result
 */
export function validateInput(text) {
  if (!text || text.trim().length === 0) {
    return { valid: false, error: 'Text cannot be empty' };
  }
  
  if (text.length > 5000) {
    return { valid: false, error: 'Text exceeds maximum length of 5000 characters' };
  }
  
  return { valid: true };
}
