# 国际化 (i18n) 使用说明

本项目已完全集成了中英双语国际化支持，使用 `react-i18next` 库实现。

## 功能特性

- 🌐 支持中文（简体）和英文
- 🔄 实时语言切换，无需刷新页面
- 💾 语言偏好自动保存到本地存储
- 🎯 智能语言检测（优先级：本地存储 > 浏览器语言 > 默认中文）
- 📱 全组件覆盖，包括所有UI文本和错误消息

## 文件结构

```
frontend/src/i18n/
├── index.js              # i18n配置文件
└── locales/
    ├── zh.json           # 中文语言资源
    └── en.json           # 英文语言资源
```

## 使用方法

### 1. 在组件中使用翻译

```javascript
import React from 'react';
import { useTranslation } from 'react-i18next';

function MyComponent() {
  const { t } = useTranslation();
  
  return (
    <div>
      <h1>{t('app.title')}</h1>
      <p>{t('app.subtitle')}</p>
    </div>
  );
}
```

### 2. 使用插值参数

```javascript
// JSON中的定义
{
  "messages": {
    "querySuccess": "查询成功！找到 {{count}} 个相似面片"
  }
}

// 组件中的使用
const message = t('messages.querySuccess', { count: 10 });
```

### 3. 语言切换

项目已包含 `LanguageSwitcher` 组件，可以直接使用：

```javascript
import LanguageSwitcher from './components/LanguageSwitcher';

function App() {
  return (
    <div>
      <LanguageSwitcher />
      {/* 其他内容 */}
    </div>
  );
}
```

### 4. 程序化语言切换

```javascript
import { useTranslation } from 'react-i18next';

function MyComponent() {
  const { i18n } = useTranslation();
  
  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    localStorage.setItem('language', lng);
  };
  
  return (
    <button onClick={() => changeLanguage('en')}>
      Switch to English
    </button>
  );
}
```

## 语言资源结构

### 主要键值分组

- `app` - 应用基本信息（标题、副标题）
- `navigation` - 导航相关（语言选择器）
- `status` - 状态信息（加载中、连接状态等）
- `stats` - 统计信息
- `query` - 查询相关
- `results` - 结果显示
- `visualization` - 可视化模态框
- `patchInfo` - 面片详细信息
- `messages` - 系统消息和通知
- `buttons` - 按钮文本

### 添加新的翻译

1. 在 `zh.json` 中添加中文键值对
2. 在 `en.json` 中添加对应的英文翻译
3. 在组件中使用 `t('key.path')` 调用

## 最佳实践

### 1. 键名命名规范
- 使用驼峰命名法：`querySuccess`
- 按功能模块分组：`query.title`、`results.loading`
- 保持键名描述性：`loadingFailed` 而不是 `error1`

### 2. 插值使用
```javascript
// 推荐：明确的参数名
"userCount": "共有 {{count}} 个用户"

// 避免：不明确的参数
"userInfo": "用户：{{0}} 年龄：{{1}}"
```

### 3. 复数处理
```javascript
// 可以使用条件表达式处理复数
"itemCount": "找到 {{count}} 个{{count, plural, =1{项目} other{项目}}}"
```

### 4. 长文本处理
对于长段落文本，建议拆分为多个键：

```json
{
  "help": {
    "step1": "第一步：选择网格文件",
    "step2": "第二步：执行查询",
    "step3": "第三步：查看结果"
  }
}
```

## 开发模式调试

开发模式下会在控制台显示i18n调试信息，包括：
- 缺失的翻译键
- 语言加载状态
- 翻译解析过程

## 添加新语言

1. 在 `locales/` 目录下创建新的语言文件（如 `fr.json`）
2. 复制现有语言文件的结构并翻译内容
3. 在 `i18n/index.js` 中添加新语言资源
4. 更新 `LanguageSwitcher` 组件添加新语言选项

## 注意事项

- 所有用户可见的文本都应该使用i18n
- 控制台日志可以保留英文，但用户提示必须国际化
- 错误消息在API层保持英文，在UI层进行国际化处理
- 定期检查翻译的一致性和准确性

## 故障排除

### 翻译不显示
1. 检查键名是否正确
2. 确认语言文件中是否存在该键
3. 查看控制台是否有i18n错误信息

### 语言切换不生效
1. 检查localStorage中的'language'值
2. 确认i18n配置是否正确加载
3. 重启开发服务器

### 新翻译不生效
1. 保存语言文件后刷新页面
2. 检查JSON格式是否正确
3. 确认键名路径是否正确
