// 主仪表板组件
import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import StatsCard from './StatsCard';
import ResultsTable from './ResultsTable';
import VisualizationModal from './VisualizationModal';
import LanguageSwitcher from './LanguageSwitcher';
import APIService from '../services/api';

function Dashboard() {
  const { t } = useTranslation();
  // 状态管理
  const [meshes, setMeshes] = useState([]);
  const [selectedMesh, setSelectedMesh] = useState('');
  const [queryResults, setQueryResults] = useState([]);
  const [queryPatchInfo, setQueryPatchInfo] = useState(null); // 新增：保存查询面片信息
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState({
    meshes: false,
    stats: false,
    query: false
  });
  const [alert, setAlert] = useState(null);
  const [apiStatus, setApiStatus] = useState({ connected: false, healthy: false });
  const [visualizationModal, setVisualizationModal] = useState({
    isOpen: false,
    dbIndex: null,
    patchInfo: null
  });

  // 组件挂载时加载初始数据
  useEffect(() => {
    initializeData();
  }, []);

  // 初始化数据
  const initializeData = async () => {
    // 并行加载数据以提高性能
    await Promise.all([
      checkApiConnection(),
      loadMeshes(),
      loadStats()
    ]);
  };

  // 检查API连接状态
  const checkApiConnection = async () => {
    try {
      const [statusData, healthData] = await Promise.all([
        APIService.getStatus(),
        APIService.healthCheck()
      ]);
      
      setApiStatus({
        connected: true,
        healthy: healthData.components_initialized || false,
        datasetSize: healthData.dataset_size || 0
      });
      
      console.log('✅', t('messages.apiConnected'));
    } catch (error) {
      console.error('❌ API连接失败:', error);
      setApiStatus({ connected: false, healthy: false });
      setAlert({ 
        type: 'error', 
        message: t('messages.apiConnectionFailed')
      });
    }
  };

  // 加载网格文件列表
  const loadMeshes = async () => {
    setLoading(prev => ({ ...prev, meshes: true }));
    
    try {
      const data = await APIService.getMeshes();
      setMeshes(data.meshes || []);
      console.log(`✅ ${t('messages.meshesLoaded', { count: data.meshes?.length || 0 })}`);
    } catch (error) {
      console.error('❌ 加载网格文件失败:', error);
      setAlert({ type: 'error', message: t('messages.loadMeshFailed', { error: error.message }) });
    } finally {
      setLoading(prev => ({ ...prev, meshes: false }));
    }
  };

  // 加载数据集统计信息
  const loadStats = async () => {
    setLoading(prev => ({ ...prev, stats: true }));
    
    try {
      const data = await APIService.getDatasetStats();
      setStats(data);
      console.log('✅', t('messages.statsLoaded'));
    } catch (error) {
      console.error('❌ 加载统计信息失败:', error);
      // 统计信息失败不显示错误，因为不是关键功能
    } finally {
      setLoading(prev => ({ ...prev, stats: false }));
    }
  };

  // 执行查询
  const handleQuery = async (maxResults = 10) => {
    if (!selectedMesh) {
      setAlert({ type: 'warning', message: t('messages.selectMeshFirst') });
      return;
    }

    setLoading(prev => ({ ...prev, query: true }));
    setAlert(null);
    setQueryResults([]);
    
    try {
      console.log(`🔍 ${t('messages.queryStarted', { mesh: selectedMesh })}`);
      const result = await APIService.performQuery(selectedMesh, maxResults);
      
      if (result.success) {
        setQueryResults(result.results);
        setQueryPatchInfo(result.query_patch_info); // 保存查询面片信息
        setAlert({ 
          type: 'success', 
          message: t('messages.querySuccess', { count: result.results.length })
        });
        console.log(`✅ ${t('messages.queryCompleted', { count: result.results.length })}`);
        console.log(`🎯 ${t('messages.queryPatchInfo')}`, result.query_patch_info);
      } else {
        setAlert({ type: 'warning', message: result.message });
      }
    } catch (error) {
      console.error('❌ 查询失败:', error);
      setAlert({ type: 'error', message: t('messages.queryFailed', { error: error.message }) });
    } finally {
      setLoading(prev => ({ ...prev, query: false }));
    }
  };

  // 处理3D可视化
  const handleVisualize = (dbIndex, patchInfo) => {
    console.log(`🎨 ${t('messages.visualizationOpened', { dbIndex })}`);
    setVisualizationModal({ 
      isOpen: true, 
      dbIndex, 
      patchInfo 
    });
  };

  // 关闭可视化模态框
  const closeVisualizationModal = () => {
    setVisualizationModal({ 
      isOpen: false, 
      dbIndex: null, 
      patchInfo: null 
    });
  };

  // 获取随机查询
  const handleRandomQuery = async () => {
    try {
      setAlert({ type: 'info', message: t('messages.randomQueryGetting') });
      const result = await APIService.getRandomQuery();
      
      if (result.success) {
        // 直接显示随机面片的可视化
        handleVisualize(result.db_index, result.patch_info);
        setAlert({ 
          type: 'success', 
          message: t('messages.randomQuerySuccess', { patternId: result.patch_info?.pattern_id })
        });
      }
    } catch (error) {
      console.error('❌ 随机查询失败:', error);
      setAlert({ type: 'error', message: t('messages.randomQueryFailed', { error: error.message }) });
    }
  };

  // 清除警告消息
  const clearAlert = () => setAlert(null);

  // 渲染警告/成功消息
  const renderAlert = () => {
    if (!alert) return null;

    const alertStyles = {
      error: { backgroundColor: '#ffebee', borderColor: '#f44336', color: '#c62828' },
      warning: { backgroundColor: '#fff3e0', borderColor: '#ff9800', color: '#ef6c00' },
      success: { backgroundColor: '#e8f5e8', borderColor: '#4caf50', color: '#2e7d32' },
      info: { backgroundColor: '#e3f2fd', borderColor: '#2196f3', color: '#1565c0' }
    };

    const style = alertStyles[alert.type] || alertStyles.info;

    return (
      <div style={{
        ...style,
        padding: '16px 20px',
        borderRadius: '8px',
        marginBottom: '24px',
        border: `1px solid ${style.borderColor}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>
            {alert.type === 'error' ? '❌' : 
             alert.type === 'warning' ? '⚠️' : 
             alert.type === 'success' ? '✅' : 'ℹ️'}
          </span>
          <span>{alert.message}</span>
        </div>
        <button
          onClick={clearAlert}
          style={{
            background: 'none',
            border: 'none',
            color: style.color,
            cursor: 'pointer',
            fontSize: '16px',
            padding: '4px'
          }}
        >
          ×
        </button>
      </div>
    );
  };

  // 主要样式
  const containerStyle = {
    minHeight: '100vh',
    backgroundColor: '#f5f7fa',
    padding: '32px',
    fontFamily: 'Arial, sans-serif'
  };

  const maxWidthContainerStyle = {
    maxWidth: '1400px',
    margin: '0 auto'
  };

  const headerStyle = {
    textAlign: 'center',
    marginBottom: '48px'
  };

  const titleStyle = {
    fontSize: '48px',
    margin: '0 0 12px 0',
    color: '#333',
    fontWeight: 'bold'
  };

  const subtitleStyle = {
    fontSize: '18px',
    color: '#666',
    margin: 0
  };

  const cardStyle = {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    marginBottom: '24px'
  };

  const sectionTitleStyle = {
    marginTop: 0,
    marginBottom: '24px',
    fontSize: '20px',
    fontWeight: '600',
    color: '#333'
  };

  const controlsGridStyle = {
    display: 'grid',
    gridTemplateColumns: '2fr auto auto',
    gap: '16px',
    alignItems: 'end'
  };

  const selectStyle = {
    width: '100%',
    padding: '12px 16px',
    borderRadius: '6px',
    border: '1px solid #ddd',
    fontSize: '16px',
    backgroundColor: 'white'
  };

  const buttonStyle = {
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    padding: '14px 24px',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: '500',
    transition: 'all 0.2s ease',
    whiteSpace: 'nowrap'
  };

  const disabledButtonStyle = {
    ...buttonStyle,
    backgroundColor: '#ccc',
    cursor: 'not-allowed'
  };

  return (
    <div style={containerStyle}>
      <div style={maxWidthContainerStyle}>
        {/* 页面标题 */}
        <div style={headerStyle}>
          {/* 语言切换器 */}
          <div style={{ 
            position: 'absolute', 
            top: '20px', 
            right: '20px', 
            zIndex: 1000 
          }}>
            <LanguageSwitcher />
          </div>
          
          <h1 style={titleStyle}>
            🎯 {t('app.title')}
          </h1>
          <p style={subtitleStyle}>
            {t('app.subtitle')}
          </p>
          
          {/* API状态指示器 */}
          <div style={{ marginTop: '16px', fontSize: '14px' }}>
            <span style={{
              padding: '4px 12px',
              borderRadius: '20px',
              backgroundColor: apiStatus.connected ? '#4caf50' : '#f44336',
              color: 'white',
              fontWeight: '500'
            }}>
              {apiStatus.connected ? `🟢 ${t('status.apiConnected')}` : `🔴 ${t('status.apiDisconnected')}`}
            </span>
            {apiStatus.connected && (
              <span style={{
                marginLeft: '12px',
                padding: '4px 12px',
                borderRadius: '20px',
                backgroundColor: apiStatus.healthy ? '#4caf50' : '#ff9800',
                color: 'white',
                fontWeight: '500'
              }}>
                {apiStatus.healthy ? `🟢 ${t('status.serviceHealthy')}` : `🟡 ${t('status.serviceUnhealthy')}`}
              </span>
            )}
          </div>
        </div>

        {/* 统计信息卡片 */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '24px',
          marginBottom: '32px'
        }}>
          <StatsCard
            title={t('stats.totalPatches')}
            value={stats?.total_patches || apiStatus.datasetSize || 0}
            icon="📊"
            color="#2196f3"
            loading={loading.stats}
          />
          <StatsCard
            title={t('stats.availableMeshes')}
            value={meshes.length}
            icon="📁"
            color="#4caf50"
            loading={loading.meshes}
          />
          <StatsCard
            title={t('stats.databaseFields')}
            value={stats?.database_columns?.length || 0}
            icon="🗃️"
            color="#ff9800"
            loading={loading.stats}
          />
          <StatsCard
            title={t('stats.queryResults')}
            value={queryResults.length}
            icon="🔍"
            color="#9c27b0"
            loading={false}
          />
        </div>

        {/* 查询控制面板 */}
        <div style={cardStyle}>
          <h2 style={sectionTitleStyle}>🔍 {t('query.title')}</h2>
          
          <div style={controlsGridStyle}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                {t('query.selectMesh')}
              </label>
              <select
                value={selectedMesh}
                onChange={(e) => setSelectedMesh(e.target.value)}
                style={selectStyle}
                disabled={loading.meshes}
              >
                <option value="">
                  {loading.meshes ? t('status.loading') : t('query.selectMeshPlaceholder')}
                </option>
                {meshes.map((mesh, index) => (
                  <option key={index} value={mesh.path}>
                    {mesh.name} ({mesh.relative_path})
                  </option>
                ))}
              </select>
            </div>
            
            <button
              style={loading.query || !selectedMesh || !apiStatus.connected ? disabledButtonStyle : buttonStyle}
              onClick={() => handleQuery(10)}
              disabled={loading.query || !selectedMesh || !apiStatus.connected}
            >
              {loading.query ? `⏳ ${t('status.querying')}` : `🔍 ${t('query.executeQuery')}`}
            </button>
            
            <button
              style={!apiStatus.connected ? disabledButtonStyle : { ...buttonStyle, backgroundColor: '#ff9800' }}
              onClick={handleRandomQuery}
              disabled={!apiStatus.connected}
            >
              🎲 {t('query.randomQuery')}
            </button>
          </div>
          
          {/* 查询提示 */}
          <div style={{
            marginTop: '16px',
            padding: '12px',
            backgroundColor: '#e3f2fd',
            borderRadius: '6px',
            fontSize: '14px',
            color: '#1565c0'
          }}>
            💡 {t('query.hint')}
          </div>
        </div>

        {/* 警告/成功消息 */}
        {renderAlert()}

        {/* 查询结果 */}
        <div style={cardStyle}>
          <h2 style={sectionTitleStyle}>📊 {t('results.title')}</h2>
          <ResultsTable 
            results={queryResults} 
            onVisualize={handleVisualize}
            loading={loading.query}
          />
        </div>

        {/* 3D可视化模态框 */}
        <VisualizationModal
          isOpen={visualizationModal.isOpen}
          onClose={closeVisualizationModal}
          dbIndex={visualizationModal.dbIndex}
          patchInfo={visualizationModal.patchInfo}
          queryPatchInfo={queryPatchInfo}
        />
      </div>
    </div>
  );
}

export default Dashboard;


