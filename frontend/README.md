# 3D网格面片可视化查询系统 - React前端

这是一个基于React的现代化前端界面，用于3D网格面片的相似性查询和可视化。

## 🚀 快速开始

### 前提条件

- Node.js (版本 14 或更高)
- npm 或 yarn
- 后端API服务运行在 http://localhost:8000

### 安装和运行

```bash
# 安装依赖
npm install

# 启动开发服务器
npm start

# 构建生产版本
npm run build

# 运行测试
npm test
```

## 📁 项目结构

```
src/
├── components/           # React组件
│   ├── Dashboard.js     # 主仪表板
│   ├── StatsCard.js     # 统计卡片
│   ├── ResultsTable.js  # 结果表格
│   └── VisualizationModal.js  # 3D可视化模态框
├── services/
│   └── api.js          # API服务层
├── App.js              # 主应用组件
├── App.css             # 应用样式
├── index.js            # 应用入口
└── index.css           # 全局样式
```

## 🎯 主要功能

### 📊 数据概览
- 实时显示数据集统计信息
- 可用网格文件列表
- API连接状态监控

### 🔍 智能查询
- 网格文件选择和查询
- 相似度排序结果
- 随机查询功能

### 🎨 3D可视化
- 交互式3D面片展示
- 详细的几何信息面板
- Plotly.js支持的高质量渲染

### 💫 用户体验
- 响应式设计
- 实时加载状态
- 错误处理和提示
- 现代化UI设计

## 🔧 配置说明

### API配置

API基础URL在 `src/services/api.js` 中配置：

```javascript
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'http://your-production-domain.com' 
  : 'http://localhost:8000';
```

### 代理配置

在 `package.json` 中设置了代理：

```json
{
  "proxy": "http://localhost:8000"
}
```

这允许前端直接调用 `/api/*` 路径的API。

## 🎨 样式系统

### CSS架构
- 全局样式：`src/index.css`
- 组件样式：内联样式 + CSS类
- 响应式设计：CSS Grid 和 Flexbox

### 设计系统
- 主色调：蓝色 (#2196f3)
- 成功色：绿色 (#4caf50)
- 警告色：橙色 (#ff9800)
- 错误色：红色 (#f44336)

## 📱 响应式设计

支持以下断点：
- 桌面：> 1024px
- 平板：768px - 1024px
- 手机：< 768px

## 🔍 调试指南

### 开发者工具
1. 打开浏览器开发者工具
2. 查看Console标签获取日志信息
3. 查看Network标签监控API请求

### 常见问题

**API连接失败**
- 检查后端服务是否运行
- 验证API URL配置
- 查看CORS设置

**3D可视化无法加载**
- 检查Plotly.js依赖
- 验证数据格式
- 查看浏览器控制台错误

**样式显示异常**
- 清除浏览器缓存
- 检查CSS导入
- 验证响应式样式

## 🧪 测试

```bash
# 运行所有测试
npm test

# 运行测试并查看覆盖率
npm test -- --coverage

# 运行特定测试文件
npm test ComponentName.test.js
```

## 📦 构建和部署

### 开发构建
```bash
npm run build
```

### 生产部署
1. 修改 `src/services/api.js` 中的生产API URL
2. 运行 `npm run build`
3. 将 `build/` 目录部署到Web服务器

### Docker部署 (可选)
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

## 🔄 与后端API的交互

### API端点
- `GET /api/meshes` - 获取网格文件列表
- `POST /api/query` - 执行相似性查询  
- `GET /api/visualize/{id}` - 获取3D可视化数据
- `GET /api/dataset/stats` - 获取数据集统计
- `GET /api/health` - 健康检查

### 数据流
1. 用户选择网格文件
2. 前端发送查询请求到API
3. 后端处理并返回相似面片列表
4. 用户点击查看按钮
5. 前端请求特定面片的3D可视化数据
6. 在模态框中展示3D可视化

## 🛠️ 开发指南

### 添加新组件
1. 在 `src/components/` 创建新的JS文件
2. 使用函数式组件和React Hooks
3. 导入必要的依赖
4. 添加适当的样式
5. 在父组件中导入和使用

### 添加新API端点
1. 在 `src/services/api.js` 中添加新方法
2. 处理错误和加载状态
3. 在组件中调用新的API方法
4. 更新UI以显示新数据

### 性能优化
- 使用 React.memo 优化组件渲染
- 实现适当的加载状态
- 使用 useMemo 和 useCallback 优化计算
- 考虑代码分割和懒加载

## 📝 待办事项

- [ ] 添加用户认证
- [ ] 实现查询历史记录
- [ ] 添加数据导出功能
- [ ] 支持多语言
- [ ] 添加更多可视化选项
- [ ] 实现离线支持
- [ ] 添加单元测试覆盖

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

此项目基于 MIT 许可证。


