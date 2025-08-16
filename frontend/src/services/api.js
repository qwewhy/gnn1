// API服务层 - 与FastAPI后端通信

import axios from 'axios';

// API基础配置
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'http://your-production-domain.com' 
  : 'http://localhost:8000';

// 创建axios实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30秒超时
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false, // 设置为false来避免CORS问题
});

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🌐 API请求: ${config.method?.toUpperCase()} ${config.url}`);
    console.log(`🌐 完整URL: ${config.baseURL}${config.url}`);
    console.log(`🌐 请求头:`, config.headers);
    return config;
  },
  (error) => {
    console.error('❌ API请求错误:', error);
    return Promise.reject(error);
  }
);

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API响应: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API响应错误:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * API服务类
 */
class APIService {
  /**
   * 获取API状态
   */
  static async getStatus() {
    try {
      const response = await apiClient.get('/');
      return response.data;
    } catch (error) {
      throw new Error(`API status error: ${error.message}`);
    }
  }

  /**
   * 健康检查
   */
  static async healthCheck() {
    try {
      const response = await apiClient.get('/api/health');
      return response.data;
    } catch (error) {
      throw new Error(`Health check error: ${error.message}`);
    }
  }

  /**
   * 获取所有可用的网格文件
   */
  static async getMeshes() {
    try {
      const response = await apiClient.get('/api/meshes');
      return response.data;
    } catch (error) {
      throw new Error(`Mesh list error: ${error.message}`);
    }
  }

  /**
   * 执行面片相似性查询
   * @param {string} meshPath - 网格文件路径
   * @param {number} maxResults - 最大结果数量
   */
  static async performQuery(meshPath, maxResults = 6) {
    try {
      const response = await apiClient.post('/api/query', {
        mesh_path: meshPath,
        max_results: maxResults
      });
      return response.data;
    } catch (error) {
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      }
      throw new Error(`Query error: ${error.message}`);
    }
  }

  /**
   * 获取特定面片的可视化数据
   * @param {number} dbIndex - 数据库索引
   */
  static async getVisualization(dbIndex) {
    try {
      const response = await apiClient.get(`/api/visualize/${dbIndex}`);
      return response.data;
    } catch (error) {
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      }
      throw new Error(`Visualization error: ${error.message}`);
    }
  }

  /**
   * 获取数据集统计信息
   */
  static async getDatasetStats() {
    try {
      const response = await apiClient.get('/api/dataset/stats');
      return response.data;
    } catch (error) {
      throw new Error(`Dataset stats error: ${error.message}`);
    }
  }

  /**
   * 获取随机查询样本
   */
  static async getRandomQuery() {
    try {
      const response = await apiClient.get('/api/query/random');
      return response.data;
    } catch (error) {
      throw new Error(`Random query error: ${error.message}`);
    }
  }

  /**
   * 获取查询面片的可视化数据
   * @param {string} meshPath - 网格文件路径
   * @param {Array} patchIndices - 面片索引数组（可选）
   */
  static async getQueryVisualization(meshPath, patchIndices = []) {
    try {
      const response = await apiClient.post('/api/visualize/query', {
        mesh_path: meshPath,
        patch_indices: patchIndices
      });
      return response.data;
    } catch (error) {
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      }
      throw new Error(`Query visualization error: ${error.message}`);
    }
  }
}

export default APIService;


