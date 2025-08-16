// 查询结果表格组件
import React from 'react';
import { useTranslation } from 'react-i18next';

function ResultsTable({ results, onVisualize, loading = false }) {
  const { t } = useTranslation();
  if (loading) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        backgroundColor: '#e3f2fd',
        borderRadius: '8px',
        border: '1px solid #2196f3'
      }}>
        <div className="loading-spinner" style={{ marginBottom: '16px' }}></div>
        <div style={{ color: '#1565c0' }}>{t('results.loading')}</div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div style={{
        padding: '40px',
        backgroundColor: '#e3f2fd',
        borderRadius: '8px',
        border: '1px solid #2196f3',
        color: '#1565c0',
        textAlign: 'center'
      }}>
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔍</div>
        <div style={{ fontSize: '18px', marginBottom: '8px' }}>{t('results.noResults')}</div>
        <div style={{ fontSize: '14px', opacity: 0.8 }}>{t('results.noResultsHint')}</div>
      </div>
    );
  }

  const tableStyle = {
    width: '100%',
    borderCollapse: 'collapse',
    backgroundColor: 'white',
    borderRadius: '8px',
    overflow: 'hidden',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  };

  const headerStyle = {
    padding: '16px 12px',
    textAlign: 'left',
    fontWeight: '600',
    borderBottom: '2px solid #ddd',
    backgroundColor: '#f8f9fa',
    color: '#495057',
    fontSize: '14px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  };

  const cellStyle = {
    padding: '14px 12px',
    borderBottom: '1px solid #eee',
    fontSize: '14px'
  };

  const buttonStyle = {
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    padding: '8px 16px',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '13px',
    transition: 'all 0.2s ease',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px'
  };

  const getQualityBadge = (quality) => {
    const isNew = quality === 1;
    return (
      <span style={{
        padding: '4px 8px',
        borderRadius: '4px',
        fontSize: '12px',
        fontWeight: '600',
        backgroundColor: isNew ? '#4caf50' : '#9e9e9e',
        color: 'white'
      }}>
        {isNew ? t('patchInfo.qualityNew') : t('patchInfo.qualityOld')}
      </span>
    );
  };

  const getSimilarityBadge = (similarity) => {
    const percentage = (similarity * 100).toFixed(1);
    const color = similarity > 0.9 ? '#4caf50' : 
                  similarity > 0.7 ? '#ff9800' : '#2196f3';
    
    return (
      <span style={{
        padding: '4px 8px',
        borderRadius: '4px',
        fontSize: '12px',
        fontWeight: '600',
        backgroundColor: color,
        color: 'white',
        fontFamily: 'monospace'
      }}>
        {similarity.toFixed(4)} ({percentage}%)
      </span>
    );
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={headerStyle}>{t('results.rank')}</th>
            <th style={headerStyle}>{t('results.patternId')}</th>
            <th style={headerStyle}>{t('results.sides')}</th>
            <th style={headerStyle}>{t('results.sourceObject')}</th>
            <th style={headerStyle}>{t('results.quality')}</th>
            <th style={headerStyle}>{t('results.similarity')}</th>
            <th style={headerStyle}>{t('results.actions')}</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, index) => (
            <tr 
              key={result.db_index} 
              style={{
                borderBottom: index === results.length - 1 ? 'none' : '1px solid #eee'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#f8f9fa';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              <td style={cellStyle}>
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  backgroundColor: result.rank <= 3 ? '#ffd700' : '#e0e0e0',
                  color: result.rank <= 3 ? '#333' : '#666',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  #{result.rank}
                </span>
              </td>
              
              <td style={cellStyle}>
                <span style={{ fontFamily: 'monospace', fontWeight: '600' }}>
                  {result.patch_info.pattern_id}
                </span>
              </td>
              
              <td style={cellStyle}>
                <span style={{ 
                  fontFamily: 'monospace',
                  backgroundColor: '#e3f2fd',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  fontSize: '13px'
                }}>
                  {result.patch_info.sides || t('patchInfo.notAvailable')}
                </span>
              </td>
              
              <td style={cellStyle}>
                <span style={{ 
                  fontSize: '13px',
                  color: '#666',
                  fontFamily: 'monospace'
                }}>
                  {result.patch_info.source_obj || t('patchInfo.unknown')}
                </span>
              </td>
              
              <td style={cellStyle}>
                {getQualityBadge(result.patch_info.quality)}
              </td>
              
              <td style={cellStyle}>
                {getSimilarityBadge(result.similarity)}
              </td>
              
              <td style={cellStyle}>
                <button
                  style={buttonStyle}
                  onClick={() => onVisualize(result.db_index, result.patch_info)}
                  onMouseEnter={(e) => {
                    e.target.style.backgroundColor = '#1976d2';
                    e.target.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.backgroundColor = '#2196f3';
                    e.target.style.transform = 'translateY(0)';
                  }}
                >
                  👁️ {t('results.view')}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* 结果统计 */}
      <div style={{
        marginTop: '16px',
        padding: '12px',
        backgroundColor: '#f8f9fa',
        borderRadius: '6px',
        fontSize: '14px',
        color: '#666',
        textAlign: 'center'
      }}>
        📊 {t('results.statistics', {
          count: results.length,
          high: results.filter(r => r.similarity > 0.9).length,
          medium: results.filter(r => r.similarity > 0.7 && r.similarity <= 0.9).length
        })}
      </div>
    </div>
  );
}

export default ResultsTable;


