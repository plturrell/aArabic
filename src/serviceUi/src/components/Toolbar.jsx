import './Toolbar.css';

function Toolbar({ onTranslate, onClear, disabled, loading, direction }) {
  return (
    <div className="toolbar" role="toolbar" aria-label="Translation actions">
      <div className="toolbar-content">
        <button
          className="toolbar-btn toolbar-btn-secondary"
          onClick={onClear}
          disabled={disabled && !loading}
          aria-label={direction === 'ar-en' ? 'مسح النص' : 'Clear text'}
          title={direction === 'ar-en' ? 'مسح' : 'Clear'}
        >
          <span aria-hidden="true">✕</span>
          <span>{direction === 'ar-en' ? 'مسح' : 'Clear'}</span>
        </button>

        <button
          className="toolbar-btn toolbar-btn-primary"
          onClick={onTranslate}
          disabled={disabled || loading}
          aria-label={direction === 'ar-en' ? 'ترجمة النص' : 'Translate text'}
          aria-busy={loading}
          title={direction === 'ar-en' ? 'ترجمة' : 'Translate'}
        >
          {loading ? (
            <>
              <span className="spinner" role="status" aria-hidden="true"></span>
              <span>{direction === 'ar-en' ? 'جاري الترجمة...' : 'Translating...'}</span>
            </>
          ) : (
            <>
              <span aria-hidden="true">→</span>
              <span>{direction === 'ar-en' ? 'ترجمة' : 'Translate'}</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}

export default Toolbar;
