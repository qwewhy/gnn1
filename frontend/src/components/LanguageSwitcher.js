// 语言切换组件
import React from 'react';
import { useTranslation } from 'react-i18next';

function LanguageSwitcher() {
  const { i18n, t } = useTranslation();

  const changeLanguage = (language) => {
    i18n.changeLanguage(language);
    localStorage.setItem('language', language);
  };

  const currentLanguage = i18n.language || 'zh';

  const buttonStyle = {
    backgroundColor: '#f8f9fa',
    border: '1px solid #dee2e6',
    borderRadius: '6px',
    padding: '8px 12px',
    margin: '0 4px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    transition: 'all 0.2s ease',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px'
  };

  const activeButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#2196f3',
    color: 'white',
    borderColor: '#2196f3'
  };

  const containerStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 16px',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    border: '1px solid #e9ecef'
  };

  const labelStyle = {
    fontSize: '14px',
    fontWeight: '500',
    color: '#495057',
    marginRight: '8px'
  };

  return (
    <div style={containerStyle}>
      <span style={labelStyle}>
        🌐 {t('navigation.language')}:
      </span>
      
      <button
        style={currentLanguage === 'zh' ? activeButtonStyle : buttonStyle}
        onClick={() => changeLanguage('zh')}
        onMouseEnter={(e) => {
          if (currentLanguage !== 'zh') {
            e.target.style.backgroundColor = '#e9ecef';
          }
        }}
        onMouseLeave={(e) => {
          if (currentLanguage !== 'zh') {
            e.target.style.backgroundColor = '#f8f9fa';
          }
        }}
      >
        🇨🇳 {t('navigation.chinese')}
      </button>
      
      <button
        style={currentLanguage === 'en' ? activeButtonStyle : buttonStyle}
        onClick={() => changeLanguage('en')}
        onMouseEnter={(e) => {
          if (currentLanguage !== 'en') {
            e.target.style.backgroundColor = '#e9ecef';
          }
        }}
        onMouseLeave={(e) => {
          if (currentLanguage !== 'en') {
            e.target.style.backgroundColor = '#f8f9fa';
          }
        }}
      >
        🇺🇸 {t('navigation.english')}
      </button>
    </div>
  );
}

export default LanguageSwitcher;
