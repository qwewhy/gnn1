// i18n国际化配置
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// 导入语言资源
import enTranslations from './locales/en.json';
import zhTranslations from './locales/zh.json';

// 配置资源
const resources = {
  en: {
    translation: enTranslations
  },
  zh: {
    translation: zhTranslations
  }
};

i18n
  // 检测用户语言
  .use(LanguageDetector)
  // 连接 react-i18next
  .use(initReactI18next)
  // 初始化 i18next
  .init({
    resources,
    
    // 默认语言设置
    fallbackLng: 'en', // 回退语言
    lng: localStorage.getItem('language') || 'zh', // 默认中文
    
    // 插值选项
    interpolation: {
      escapeValue: false, // React已经默认转义
    },
    
    // 语言检测选项
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      lookupLocalStorage: 'language',
    },
    
    // 开发模式调试
    debug: process.env.NODE_ENV === 'development',
    
    // 命名空间
    defaultNS: 'translation',
    ns: ['translation'],
  });

export default i18n;
