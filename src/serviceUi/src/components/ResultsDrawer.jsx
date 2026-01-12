import './ResultsDrawer.css';

function ResultsDrawer({ show, results, loading, onClose, direction }) {
  if (!show) return null;

  return (
    <>
      <div 
        className="drawer-backdrop" 
        onClick={onClose}
        role="presentation"
        aria-hidden="true"
      />
      <div 
        className={`results-drawer ${show ? 'open' : ''}`}
        role="dialog"
        aria-label={direction === 'ar-en' ? 'نتيجة الترجمة' : 'Translation Result'}
        aria-modal="true"
      >
        <div className="drawer-header">
          <h2 className="drawer-title" id="drawer-title">
            {direction === 'ar-en' ? 'نتيجة الترجمة' : 'Translation Result'}
          </h2>
          <button 
            className="drawer-close" 
            onClick={onClose}
            aria-label="Close translation results"
            title="Close"
          >
            ×
          </button>
        </div>

        <div className="drawer-content">
          {loading ? (
            <div className="drawer-loading" role="status" aria-live="polite">
              <span className="spinner-large" aria-hidden="true"></span>
              <p>{direction === 'ar-en' ? 'جاري الترجمة...' : 'Translating...'}</p>
            </div>
          ) : results?.error ? (
            <div className="drawer-error" role="alert">
              <span className="error-icon" aria-hidden="true">⚠️</span>
              <p className="error-message">{results.error}</p>
            </div>
          ) : results?.translation ? (
            <div className="translation-result">
              <div className="result-section">
                <h3 className="section-title">
                  {direction === 'ar-en' ? 'النص الأصلي' : 'Original Text'}
                </h3>
                <div 
                  className="result-text source-text"
                  dir={direction === 'ar-en' ? 'rtl' : 'ltr'}
                  lang={direction === 'ar-en' ? 'ar' : 'en'}
                >
                  {results.source}
                </div>
              </div>

              <div className="result-divider" aria-hidden="true">→</div>

              <div className="result-section">
                <h3 className="section-title">
                  {direction === 'ar-en' ? 'الترجمة' : 'Translation'}
                </h3>
                <div 
                  className="result-text translated-text"
                  dir={direction === 'ar-en' ? 'ltr' : 'rtl'}
                  lang={direction === 'ar-en' ? 'en' : 'ar'}
                >
                  {results.translation}
                </div>
                <button 
                  className="copy-btn"
                  onClick={() => {
                    navigator.clipboard.writeText(results.translation);
                    // Optional: Show feedback toast
                  }}
                  aria-label={direction === 'ar-en' ? 'نسخ الترجمة' : 'Copy translation'}
                  title={direction === 'ar-en' ? 'نسخ' : 'Copy'}
                >
                  <span aria-hidden="true">📋</span> {direction === 'ar-en' ? 'نسخ' : 'Copy'}
                </button>
              </div>
            </div>
          ) : (
            <div className="drawer-empty">
              <p>{direction === 'ar-en' ? 'لا توجد نتائج' : 'No results'}</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default ResultsDrawer;
