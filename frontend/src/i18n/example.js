// i18n使用示例
import React from 'react';
import { useTranslation } from 'react-i18next';

function I18nExample() {
  const { t, i18n } = useTranslation();

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    localStorage.setItem('language', lng);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>{t('app.title')}</h1>
      <p>{t('app.subtitle')}</p>
      
      <div style={{ marginTop: '20px' }}>
        <h3>语言切换示例：</h3>
        <button 
          onClick={() => changeLanguage('zh')}
          style={{ 
            marginRight: '10px',
            padding: '8px 16px',
            backgroundColor: i18n.language === 'zh' ? '#2196f3' : '#f0f0f0',
            color: i18n.language === 'zh' ? 'white' : 'black',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          中文
        </button>
        <button 
          onClick={() => changeLanguage('en')}
          style={{ 
            padding: '8px 16px',
            backgroundColor: i18n.language === 'en' ? '#2196f3' : '#f0f0f0',
            color: i18n.language === 'en' ? 'white' : 'black',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          English
        </button>
      </div>

      <div style={{ marginTop: '20px' }}>
        <h3>翻译示例：</h3>
        <ul>
          <li><strong>加载状态：</strong> {t('status.loading')}</li>
          <li><strong>查询按钮：</strong> {t('query.executeQuery')}</li>
          <li><strong>关闭按钮：</strong> {t('buttons.close')}</li>
          <li><strong>统计信息：</strong> {t('stats.totalPatches')}</li>
        </ul>
      </div>

      <div style={{ marginTop: '20px' }}>
        <h3>插值示例：</h3>
        <p>{t('messages.querySuccess', { count: 5 })}</p>
        <p>{t('results.statistics', { count: 10, high: 3, medium: 4 })}</p>
      </div>

      <div style={{ marginTop: '20px' }}>
        <h3>当前语言：</h3>
        <p>
          <strong>{i18n.language === 'zh' ? '中文' : 'English'}</strong>
          {' '}(代码: {i18n.language})
        </p>
      </div>
    </div>
  );
}

export default I18nExample;
