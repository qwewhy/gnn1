// 统计卡片组件
import React from 'react';
import { useTranslation } from 'react-i18next';

function StatsCard({ title, value, icon, color = '#2196f3', loading = false }) {
  const { t } = useTranslation();
  const cardStyle = {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    border: `2px solid ${color}20`,
    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
    cursor: loading ? 'default' : 'pointer'
  };

  const iconStyle = {
    fontSize: '32px',
    filter: loading ? 'grayscale(100%)' : 'none',
    opacity: loading ? 0.5 : 1
  };

  const valueStyle = {
    fontSize: '32px',
    fontWeight: 'bold',
    color: loading ? '#ccc' : color,
    fontFamily: 'monospace'
  };

  const titleStyle = {
    fontSize: '14px',
    color: loading ? '#999' : '#666',
    margin: 0,
    fontWeight: '500'
  };

  return (
    <div 
      style={cardStyle}
      onMouseEnter={(e) => {
        if (!loading) {
          e.target.style.transform = 'translateY(-2px)';
          e.target.style.boxShadow = '0 6px 12px rgba(0, 0, 0, 0.15)';
        }
      }}
      onMouseLeave={(e) => {
        if (!loading) {
          e.target.style.transform = 'translateY(0)';
          e.target.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
        }
      }}
    >
      <div style={iconStyle}>
        {loading ? '⏳' : icon}
      </div>
      <div>
        <div style={valueStyle}>
          {loading ? '...' : value}
        </div>
        <div style={titleStyle}>{title}</div>
      </div>
    </div>
  );
}

export default StatsCard;


