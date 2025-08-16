// 3D可视化模态框组件
import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import Plot from 'react-plotly.js';
import APIService from '../services/api';

function VisualizationModal({ isOpen, onClose, dbIndex, patchInfo, queryPatchInfo }) {
  const { t } = useTranslation();
  const [visualization, setVisualization] = useState(null);
  const [queryVisualization, setQueryVisualization] = useState(null);
  const [loading, setLoading] = useState(false);
  const [queryLoading, setQueryLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queryError, setQueryError] = useState(null);

  useEffect(() => {
    if (isOpen && dbIndex !== null) {
      loadVisualization();
      // 如果有查询面片信息，也加载查询面片可视化
      if (queryPatchInfo) {
        loadQueryVisualization();
      }
    } else {
      // 重置状态
      setVisualization(null);
      setQueryVisualization(null);
      setError(null);
      setQueryError(null);
    }
  }, [isOpen, dbIndex, queryPatchInfo]);

  const loadVisualization = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log(`🎨 ${t('messages.visualizationOpened', { dbIndex })}`); // 保留英文log作为备注
      const data = await APIService.getVisualization(dbIndex);
      
      if (data.success) {
        setVisualization(data);
        console.log('✅', t('messages.visualizationLoaded'));
      } else {
        setError(data.message || '可视化加载失败');
      }
    } catch (error) {
      console.error('❌ 可视化加载失败:', error);
      setError(error.message || '可视化加载失败');
    } finally {
      setLoading(false);
    }
  };

  const loadQueryVisualization = async () => {
    setQueryLoading(true);
    setQueryError(null);
    
    try {
      console.log('🎨', t('messages.queryVisualizationLoadFailed')); // 抄错了，应该是加载中
      const data = await APIService.getQueryVisualization(
        queryPatchInfo.mesh_path, 
        queryPatchInfo.patch_indices
      );
      
      if (data.success) {
        setQueryVisualization(data);
        console.log('✅', t('messages.queryVisualizationLoaded'));
      } else {
        setQueryError(data.message || '查询面片可视化加载失败');
      }
    } catch (error) {
      console.error('❌ 查询面片可视化加载失败:', error);
      setQueryError(error.message || '查询面片可视化加载失败');
    } finally {
      setQueryLoading(false);
    }
  };

  if (!isOpen) return null;

  const modalOverlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
    padding: '20px'
  };

  const modalContentStyle = {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '24px',
    maxWidth: '98vw', // 增加宽度到98%
    width: '1600px', // 设置固定宽度，在大屏幕上更宽
    maxHeight: '95vh',
    overflow: 'auto',
    position: 'relative',
    boxShadow: '0 10px 25px rgba(0,0,0,0.2)'
  };

  const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
    paddingBottom: '16px',
    borderBottom: '1px solid #eee'
  };

  const titleStyle = {
    margin: 0,
    fontSize: '20px',
    fontWeight: '600',
    color: '#333'
  };

  const closeButtonStyle = {
    backgroundColor: '#f44336',
    color: 'white',
    border: 'none',
    padding: '8px 16px',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500'
  };

  const renderPatchInfoPanel = (patchInfo, isQuery = false) => {
    if (!patchInfo) return null;

    const panelStyle = {
      marginTop: '20px',
      padding: isQuery ? '12px' : '20px',
      backgroundColor: isQuery ? '#fff8e1' : '#f8f9fa',
      borderRadius: '8px',
      border: `1px solid ${isQuery ? '#ffcc02' : '#e9ecef'}`
    };

    const infoGridStyle = {
      display: 'grid',
      gridTemplateColumns: isQuery ? 
        'repeat(auto-fit, minmax(180px, 1fr))' : // 查询面片时使用更紧凑的布局
        '1fr 1.2fr 1fr', // 主显示时：基本+几何 | 拓扑(更宽) | 曲率+统计
      gap: '20px',
      marginTop: '16px'
    };

    const infoItemStyle = {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '8px 0',
      borderBottom: '1px solid #e0e0e0'
    };

    const labelStyle = {
      fontWeight: '500',
      color: '#495057',
      fontSize: '13px'
    };

    const valueStyle = {
      color: '#6c757d',
      fontFamily: 'monospace',
      fontSize: '13px',
      fontWeight: '600'
    };

    const numericValueStyle = {
      ...valueStyle,
      color: '#28a745'
    };

    return (
      <div style={panelStyle}>
        <h3 style={{ 
          margin: '0 0 12px 0', 
          fontSize: isQuery ? '14px' : '16px', 
          color: isQuery ? '#ff9800' : '#007bff' 
        }}>
          {isQuery ? `🔍 ${t('visualization.queryPatchInfo')}` : `📋 ${t('visualization.patchInfo')}`}
        </h3>
        
        <div style={infoGridStyle}>
          {/* 第一列：基本信息 + 几何信息 */}
          <div>
            {/* 基本信息 */}
            <h4 style={{ 
              fontSize: isQuery ? '12px' : '14px', 
              color: isQuery ? '#ff9800' : '#007bff', 
              marginBottom: '12px' 
            }}>{t('patchInfo.basicInfo')}</h4>
            <div style={infoItemStyle}>
              <span style={labelStyle}>{t('patchInfo.patternId')}</span>
              <span style={valueStyle}>{patchInfo.pattern_id}</span>
            </div>
            <div style={infoItemStyle}>
              <span style={labelStyle}>{t('patchInfo.sides')}</span>
              <span style={numericValueStyle}>{patchInfo.sides || t('patchInfo.notAvailable')}</span>
            </div>
            <div style={infoItemStyle}>
              <span style={labelStyle}>{t('patchInfo.sourceObj')}</span>
              <span style={valueStyle}>{patchInfo.source_obj || t('patchInfo.unknown')}</span>
            </div>
            <div style={infoItemStyle}>
              <span style={labelStyle}>{t('patchInfo.quality')}</span>
              <span style={{
                ...valueStyle,
                color: patchInfo.quality === 1 ? '#28a745' : '#6c757d'
              }}>
                {patchInfo.quality === 1 ? t('patchInfo.qualityNew') : t('patchInfo.qualityOld')}
              </span>
            </div>

            {/* 几何信息 */}
            <h4 style={{ fontSize: '14px', color: '#007bff', marginBottom: '12px', marginTop: '20px' }}>{t('patchInfo.geometryInfo')}</h4>
            <div style={infoItemStyle}>
              <span style={labelStyle}>{t('patchInfo.numVertices')}</span>
              <span style={numericValueStyle}>{patchInfo.num_vertices || t('patchInfo.notAvailable')}</span>
            </div>
            <div style={infoItemStyle}>
              <span style={labelStyle}>{t('patchInfo.numFaces')}</span>
              <span style={numericValueStyle}>{patchInfo.num_faces || t('patchInfo.notAvailable')}</span>
            </div>
            <div style={infoItemStyle}>
              <span style={labelStyle}>{t('patchInfo.area')}</span>
              <span style={numericValueStyle}>
                {patchInfo.area ? patchInfo.area.toFixed(4) : t('patchInfo.notAvailable')}
              </span>
            </div>
            <div style={infoItemStyle}>
              <span style={labelStyle}>{t('patchInfo.complexityScore')}</span>
              <span style={numericValueStyle}>
                {patchInfo.complexity_score ? patchInfo.complexity_score.toFixed(4) : t('patchInfo.notAvailable')}
              </span>
            </div>
            {patchInfo.total_boundary_length && (
              <div style={infoItemStyle}>
                <span style={labelStyle}>{t('patchInfo.totalBoundaryLength')}</span>
                <span style={numericValueStyle}>
                  {patchInfo.total_boundary_length.toFixed(4)}
                </span>
              </div>
            )}
          </div>

          {/* 第二列：拓扑信息（给更多空间） */}
          {(patchInfo.edgebreaker_encoding || patchInfo.canonical_form) && (
            <div>
              <h4 style={{ fontSize: '14px', color: '#007bff', marginBottom: '12px' }}>{t('patchInfo.topologyInfo')}</h4>
              {patchInfo.edgebreaker_encoding && (
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ ...labelStyle, marginBottom: '6px' }}>{t('patchInfo.edgebreakerEncoding')}</div>
                  <div style={{
                    ...valueStyle,
                    wordBreak: 'break-all',
                    whiteSpace: 'pre-wrap',
                    backgroundColor: '#f8f9fa',
                    padding: '8px',
                    borderRadius: '4px',
                    border: '1px solid #e9ecef',
                    fontFamily: 'monospace',
                    fontSize: '12px',
                    lineHeight: '1.4',
                    maxHeight: '80px',
                    overflowY: 'auto'
                  }}>
                    {patchInfo.edgebreaker_encoding}
                  </div>
                </div>
              )}
              {patchInfo.canonical_form && (
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ ...labelStyle, marginBottom: '6px' }}>{t('patchInfo.canonicalForm')}</div>
                  <div style={{
                    ...valueStyle,
                    wordBreak: 'break-all',
                    whiteSpace: 'pre-wrap',
                    backgroundColor: '#f8f9fa',
                    padding: '8px',
                    borderRadius: '4px',
                    border: '1px solid #e9ecef',
                    fontFamily: 'monospace',
                    fontSize: '12px',
                    lineHeight: '1.4',
                    maxHeight: '80px',
                    overflowY: 'auto'
                  }}>
                    {patchInfo.canonical_form}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 第三列：曲率信息 + 几何数据统计 */}
          <div>
            {/* 曲率信息 */}
            {(patchInfo.avg_curvature || patchInfo.curvature_variance) && (
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ fontSize: '14px', color: '#007bff', marginBottom: '12px' }}>{t('patchInfo.curvatureInfo')}</h4>
                {patchInfo.avg_curvature && (
                  <div style={infoItemStyle}>
                    <span style={labelStyle}>{t('patchInfo.avgCurvature')}</span>
                    <span style={numericValueStyle}>{patchInfo.avg_curvature.toFixed(4)}</span>
                  </div>
                )}
                {patchInfo.curvature_variance && (
                  <div style={infoItemStyle}>
                    <span style={labelStyle}>{t('patchInfo.curvatureVariance')}</span>
                    <span style={numericValueStyle}>{patchInfo.curvature_variance.toFixed(4)}</span>
                  </div>
                )}
              </div>
            )}

            {/* 几何数据统计 */}
            {patchInfo.geometry && Object.keys(patchInfo.geometry).length > 0 && (
              <div>
                <h4 style={{ fontSize: '14px', color: '#007bff', marginBottom: '12px' }}>{t('patchInfo.geometryStats')}</h4>
                {Object.entries(patchInfo.geometry).map(([key, value]) => (
                  <div key={key} style={infoItemStyle}>
                    <span style={labelStyle}>{key}</span>
                    <span style={valueStyle}>
                      {Array.isArray(value) ? t('patchInfo.count', { count: value.length }) : t('patchInfo.notAvailable')}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={modalOverlayStyle} onClick={onClose}>
      <div style={modalContentStyle} onClick={(e) => e.stopPropagation()}>
        <div style={headerStyle}>
          <h2 style={titleStyle}>
            🎨 {t('visualization.title')} - {t('visualization.dbIndex', { index: dbIndex })}
          </h2>
          <button
            onClick={onClose}
            style={closeButtonStyle}
            onMouseEnter={(e) => e.target.style.backgroundColor = '#d32f2f'}
            onMouseLeave={(e) => e.target.style.backgroundColor = '#f44336'}
          >
            ❌ {t('visualization.close')}
          </button>
        </div>
        
        {loading && (
          <div style={{
            textAlign: 'center',
            padding: '60px 40px',
            backgroundColor: '#f8f9fa',
            borderRadius: '8px',
            border: '1px solid #e9ecef'
          }}>
            <div className="loading-spinner" style={{ 
              width: '40px', 
              height: '40px',
              borderWidth: '4px',
              marginBottom: '20px'
            }}></div>
            <div style={{ fontSize: '16px', color: '#6c757d' }}>{t('visualization.loading')}</div>
            <div style={{ fontSize: '14px', color: '#adb5bd', marginTop: '8px' }}>
              {t('visualization.loadingTime')}
            </div>
          </div>
        )}
        
        {error && (
          <div style={{
            padding: '24px',
            backgroundColor: '#ffebee',
            borderRadius: '8px',
            border: '1px solid #f44336',
            color: '#c62828',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>❌</div>
            <div style={{ fontSize: '16px', fontWeight: '500', marginBottom: '8px' }}>
              {t('visualization.loadingFailed')}
            </div>
            <div style={{ fontSize: '14px', opacity: 0.8 }}>
              {error}
            </div>
            <button
              onClick={loadVisualization}
              style={{
                marginTop: '16px',
                backgroundColor: '#2196f3',
                color: 'white',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              🔄 {t('status.retry')}
            </button>
          </div>
        )}
        
        {/* 双面板显示区域 */}
        {queryPatchInfo && (loading || queryLoading || visualization?.success || queryVisualization?.success) && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: window.innerWidth < 1200 ? '1fr' : '1fr 1fr', // 小屏幕时单列显示
            gap: '20px',
            marginTop: '20px'
          }}>
            {/* 左侧：查询面片 */}
            <div style={{
              border: '2px solid #ff9800',
              borderRadius: '12px',
              padding: '16px',
              backgroundColor: '#fff3e0'
            }}>
              <h3 style={{
                margin: '0 0 16px 0',
                fontSize: '16px',
                color: '#ff9800',
                textAlign: 'center',
                fontWeight: '600'
              }}>
                🔍 查询面片 (来自 {queryPatchInfo.mesh_name})
              </h3>
              
              {queryLoading && (
                <div style={{
                  textAlign: 'center',
                  padding: '40px 20px',
                  backgroundColor: '#fff',
                  borderRadius: '8px'
                }}>
                  <div className="loading-spinner" style={{ marginBottom: '16px' }}></div>
                  <div style={{ color: '#666' }}>正在生成查询面片可视化...</div>
                </div>
              )}
              
              {queryError && (
                <div style={{
                  padding: '20px',
                  backgroundColor: '#ffebee',
                  borderRadius: '8px',
                  border: '1px solid #f44336',
                  color: '#c62828',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '24px', marginBottom: '8px' }}>❌</div>
                  <div style={{ fontSize: '14px' }}>{queryError}</div>
                  <button
                    onClick={loadQueryVisualization}
                    style={{
                      marginTop: '12px',
                      backgroundColor: '#2196f3',
                      color: 'white',
                      border: 'none',
                      padding: '6px 12px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    🔄 {t('status.retry')}
                  </button>
                </div>
              )}
              
              {queryVisualization?.success && (
                <div>
                  <div style={{
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    backgroundColor: 'white'
                  }}>
                    <Plot
                      data={queryVisualization.plotly_json.data}
                      layout={{
                        ...queryVisualization.plotly_json.layout,
                        height: 500, // 增加高度
                        width: window.innerWidth < 1200 ? 
                          Math.min(800, window.innerWidth - 100) : // 小屏幕时单列宽度
                          Math.min(650, (window.innerWidth - 150) / 2), // 大屏幕时双列宽度
                        margin: { l: 30, r: 30, t: 40, b: 30 }
                      }}
                      config={{ 
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['sendDataToCloud'],
                        displaylogo: false
                      }}
                    />
                  </div>
                  {queryVisualization.patch_info && (
                    <div style={{ marginTop: '12px' }}>
                      {renderPatchInfoPanel(queryVisualization.patch_info, true)}
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {/* 右侧：相似面片 */}
            <div style={{
              border: '2px solid #2196f3',
              borderRadius: '12px',
              padding: '16px',
              backgroundColor: '#e3f2fd'
            }}>
              <h3 style={{
                margin: '0 0 16px 0',
                fontSize: '16px',
                color: '#2196f3',
                textAlign: 'center',
                fontWeight: '600'
              }}>
                📊 相似面片 (DB索引: {dbIndex})
              </h3>
              
              {loading && (
                <div style={{
                  textAlign: 'center',
                  padding: '40px 20px',
                  backgroundColor: '#fff',
                  borderRadius: '8px'
                }}>
                  <div className="loading-spinner" style={{ marginBottom: '16px' }}></div>
                  <div style={{ color: '#666' }}>正在生成相似面片可视化...</div>
                </div>
              )}
              
              {error && (
                <div style={{
                  padding: '20px',
                  backgroundColor: '#ffebee',
                  borderRadius: '8px',
                  border: '1px solid #f44336',
                  color: '#c62828',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '24px', marginBottom: '8px' }}>❌</div>
                  <div style={{ fontSize: '14px' }}>{error}</div>
                  <button
                    onClick={loadVisualization}
                    style={{
                      marginTop: '12px',
                      backgroundColor: '#2196f3',
                      color: 'white',
                      border: 'none',
                      padding: '6px 12px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    🔄 {t('status.retry')}
                  </button>
                </div>
              )}
              
              {visualization?.success && (
                <div>
                  <div style={{
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    overflow: 'hidden',
                    backgroundColor: 'white'
                  }}>
                    <Plot
                      data={visualization.plotly_json.data}
                      layout={{
                        ...visualization.plotly_json.layout,
                        height: 500, // 增加高度
                        width: window.innerWidth < 1200 ? 
                          Math.min(800, window.innerWidth - 100) : // 小屏幕时单列宽度
                          Math.min(650, (window.innerWidth - 150) / 2), // 大屏幕时双列宽度
                        margin: { l: 30, r: 30, t: 40, b: 30 }
                      }}
                      config={{ 
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['sendDataToCloud'],
                        displaylogo: false
                      }}
                    />
                  </div>
                  {visualization.patch_info && (
                    <div style={{ marginTop: '12px' }}>
                      {renderPatchInfoPanel(visualization.patch_info, false)}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* 如果没有查询面片信息，则显示单一面片（兼容原有功能） */}
        {!queryPatchInfo && visualization && visualization.success && (
          <div>
            {/* 3D可视化图表 */}
            <div style={{
              border: '1px solid #e0e0e0',
              borderRadius: '8px',
              overflow: 'hidden',
              backgroundColor: 'white'
            }}>
              <Plot
                data={visualization.plotly_json.data}
                layout={{
                  ...visualization.plotly_json.layout,
                  height: 700, // 增加高度
                  width: Math.min(1400, window.innerWidth - 100), // 增加宽度
                  margin: { l: 50, r: 50, t: 60, b: 50 }
                }}
                config={{ 
                  responsive: true,
                  displayModeBar: true,
                  modeBarButtonsToRemove: ['sendDataToCloud'],
                  displaylogo: false
                }}
              />
            </div>
            
            {/* 面片信息面板 */}
            {visualization.patch_info && renderPatchInfoPanel(visualization.patch_info)}
          </div>
        )}
      </div>
    </div>
  );
}

export default VisualizationModal;


