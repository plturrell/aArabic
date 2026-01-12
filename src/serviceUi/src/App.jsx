import { useState, useEffect } from 'react';
import Canvas from './components/Canvas';
import Toolbar from './components/Toolbar';
import ResultsDrawer from './components/ResultsDrawer';
import { translateText, detectLanguage } from './services/translationService';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [direction, setDirection] = useState('ar-en'); // 'ar-en' or 'en-ar'
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);

  // Keyboard navigation - Escape key closes drawer (Apple HIG)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        if (showResults) {
          setShowResults(false);
        } else if (showSidebar) {
          setShowSidebar(false);
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [showResults, showSidebar]);

  const handleTranslate = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setShowResults(true);

    try {
      // Auto-detect language and adjust direction if needed
      const detectedLang = detectLanguage(text);
      let translationDirection = direction;
      
      if (detectedLang === 'ar' && direction === 'en-ar') {
        translationDirection = 'ar-en';
        setDirection('ar-en');
      } else if (detectedLang === 'en' && direction === 'ar-en') {
        translationDirection = 'en-ar';
        setDirection('en-ar');
      }

      const result = await translateText(text, translationDirection);
      setResults(result);
    } catch (error) {
      setResults({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResults(null);
    setShowResults(false);
  };

  const toggleDirection = () => {
    setDirection(prev => prev === 'ar-en' ? 'en-ar' : 'ar-en');
    setText('');
    setResults(null);
    setShowResults(false);
  };

  return (
    <div className="app" dir={direction === 'ar-en' ? 'rtl' : 'ltr'}>
      {/* Hamburger Menu Button */}
      <button 
        className="hamburger-btn"
        onClick={() => setShowSidebar(!showSidebar)}
        aria-label="Menu"
        aria-expanded={showSidebar}
      >
        <span className="hamburger-icon">
          <span></span>
          <span></span>
          <span></span>
        </span>
      </button>

      {/* Sidebar */}
      {showSidebar && (
        <>
          <div 
            className="sidebar-backdrop" 
            onClick={() => setShowSidebar(false)}
            role="presentation"
            aria-hidden="true"
          />
          <div className="sidebar" role="dialog" aria-label="Settings menu">
            <div className="sidebar-header">
              <h2 className="sidebar-title">
                {direction === 'ar-en' ? 'الإعدادات' : 'Settings'}
              </h2>
              <button 
                className="sidebar-close" 
                onClick={() => setShowSidebar(false)}
                aria-label="Close menu"
              >
                ×
              </button>
            </div>
            
            <div className="sidebar-content">
              <div className="sidebar-section">
                <h3 className="sidebar-section-title">
                  {direction === 'ar-en' ? 'اتجاه الترجمة' : 'Translation Direction'}
                </h3>
                <div className="language-direction-toggle">
                  <button
                    className={`direction-btn ${direction === 'ar-en' ? 'active' : ''}`}
                    onClick={() => {
                      toggleDirection();
                      setShowSidebar(false);
                    }}
                    aria-pressed={direction === 'ar-en'}
                    aria-label="Arabic to English"
                  >
                    AR → EN
                  </button>
                  <button
                    className={`direction-btn ${direction === 'en-ar' ? 'active' : ''}`}
                    onClick={() => {
                      if (direction === 'ar-en') toggleDirection();
                      setShowSidebar(false);
                    }}
                    aria-pressed={direction === 'en-ar'}
                    aria-label="English to Arabic"
                  >
                    EN → AR
                  </button>
                </div>
              </div>

              <div className="sidebar-section">
                <h3 className="sidebar-section-title">
                  {direction === 'ar-en' ? 'حول' : 'About'}
                </h3>
                <p className="sidebar-text">
                  {direction === 'ar-en' 
                    ? 'مترجم عربي ↔ إنجليزي بتقنية الذكاء الاصطناعي'
                    : 'AI-powered Arabic ↔ English translator'}
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      <main className="app-main" role="main">
        <Canvas
          value={text}
          onChange={setText}
          disabled={loading}
          direction={direction}
        />

        <Toolbar
          onTranslate={handleTranslate}
          onClear={handleClear}
          disabled={!text.trim() || loading}
          loading={loading}
          direction={direction}
        />
      </main>

      <ResultsDrawer
        show={showResults}
        results={results}
        loading={loading}
        onClose={() => setShowResults(false)}
        direction={direction}
      />
    </div>
  );
}

export default App;
