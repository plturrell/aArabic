import { useRef, useEffect } from 'react';
import './Canvas.css';

function Canvas({ value, onChange, disabled, direction }) {
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [value]);

  return (
    <div className="canvas" role="region" aria-label="Text input area">
      <div className="canvas-content">
        <label htmlFor="translation-input" className="visually-hidden">
          {direction === 'ar-en' 
            ? 'أدخل النص العربي للترجمة' 
            : 'Enter English text to translate'}
        </label>
        <textarea
          id="translation-input"
          ref={textareaRef}
          className="canvas-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={direction === 'ar-en' ? "أدخل النص العربي للترجمة..." : "Enter English text to translate..."}
          disabled={disabled}
          autoFocus
          dir={direction === 'ar-en' ? 'rtl' : 'ltr'}
          lang={direction === 'ar-en' ? 'ar' : 'en'}
          aria-label={direction === 'ar-en' ? 'منطقة إدخال النص' : 'Text input area'}
          spellCheck={true}
        />
      </div>
    </div>
  );
}

export default Canvas;
