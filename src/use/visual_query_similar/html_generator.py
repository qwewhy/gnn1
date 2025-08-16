# File: src/use/visual_query_similar/html_generator.py
# HTML生成模块 / HTML generation module

import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple


class HTMLGenerator:
    """HTML生成器 / HTML Generator"""
    
    def generate_dashboard_html(self, main_fig: go.Figure, detail_views: Dict[str, go.Figure], 
                              results: List[Tuple[int, float]], mesh_path: str, patch_infos: Dict[str, Dict] = None) -> str:
        """
        生成交互式仪表板的HTML / Generate interactive dashboard HTML
        
        Args:
            main_fig: 主图表 / Main figure
            detail_views: 详细视图字典 / Detail views dictionary
            results: 查询结果 / Query results
            mesh_path: 网格文件路径 / Mesh file path
            
        Returns:
            HTML内容字符串 / HTML content string
        """
        # 转换图表为HTML / Convert figures to HTML
        main_html = main_fig.to_html(include_plotlyjs='cdn', div_id='main-chart')
        
        detail_htmls = {}
        for key, fig in detail_views.items():
            detail_htmls[key] = fig.to_html(include_plotlyjs=False, div_id=f'detail-{key}')
        
        # 创建完整的HTML页面 / Create complete HTML page
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🎯 面片相似性查询结果 - {Path(mesh_path).name}</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 面片相似性查询结果</h1>
            <h2>📄 {Path(mesh_path).name}</h2>
            <p>💡 点击下方按钮查看详细视图，最多可同时显示两个放大视图</p>
        </div>

        <div class="stats">
            {self._generate_stats_html(results)}
        </div>

        <div class="main-chart">
            <h3>📊 概览视图</h3>
            {main_html.split('<body>')[1].split('</body>')[0]}
        </div>

        <div class="controls">
            <button class="btn" onclick="showDetail('query')">🔍 放大查询面片</button>
            <button class="btn" onclick="showDetail('similar_0')">📊 放大相似面片 1</button>
            <button class="btn" onclick="showDetail('similar_1')">📊 放大相似面片 2</button>
            <button class="btn" onclick="showDetail('similar_2')">📊 放大相似面片 3</button>
            <button class="btn" onclick="hideAllDetails()">❌ 关闭所有详细视图</button>
        </div>

        {self._generate_detail_sections_html(detail_htmls, patch_infos)}
    </div>

    <script>
        {self._get_javascript_code()}
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _get_css_styles(self) -> str:
        """获取CSS样式 / Get CSS styles"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-chart {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        .detail-section {
            display: none;
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .detail-content {
            display: flex;
            gap: 20px;
            min-height: 600px;
        }
        .detail-viewer {
            flex: 2;
            min-width: 0;
        }
        .detail-info {
            flex: 1;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            overflow-y: auto;
            max-height: 600px;
        }
        .info-group {
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .info-title {
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .info-item {
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .info-label {
            font-weight: 500;
            color: #495057;
            font-size: 13px;
        }
        .info-value {
            color: #6c757d;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-width: 60%;
            text-align: right;
            word-wrap: break-word;
        }
        .info-value.numeric {
            color: #28a745;
            font-weight: bold;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            margin: 0 10px;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        .btn:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }
        .btn.active {
            background: #28a745;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        """
    
    def _generate_stats_html(self, results: List[Tuple[int, float]]) -> str:
        """生成统计信息HTML / Generate statistics HTML"""
        if not results:
            return ""
            
        return f"""
            <div class="stat-card">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">找到相似面片</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{results[0][1]:.3f}</div>
                <div class="stat-label">最高相似度</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len([r for r in results if r[1] > 0.9])}</div>
                <div class="stat-label">高相似度面片 (>0.9)</div>
            </div>
        """
    
    def _generate_detail_sections_html(self, detail_htmls: Dict[str, str], patch_infos: Dict[str, Dict] = None) -> str:
        """生成详细视图区域HTML / Generate detail sections HTML"""
        sections_html = []
        patch_infos = patch_infos or {}
        
        # 查询面片详细视图 / Query patch detail view
        if 'query' in detail_htmls:
            query_info_html = self._generate_patch_info_panel(patch_infos.get('query', {}), is_query=True)
            sections_html.append(f"""
        <div id="detail-query" class="detail-section">
            <h3>🔍 查询面片 - 详细视图</h3>
            <div class="detail-content">
                <div class="detail-viewer">
                    {detail_htmls['query'].split('<body>')[1].split('</body>')[0]}
                </div>
                <div class="detail-info">
                    {query_info_html}
                </div>
            </div>
        </div>
            """)
        
        # 相似面片详细视图 / Similar patches detail views
        for i in range(5):  # 最多5个相似面片 / Maximum 5 similar patches
            key = f'similar_{i}'
            if key in detail_htmls:
                similar_info_html = self._generate_patch_info_panel(patch_infos.get(key, {}), is_query=False)
                sections_html.append(f"""
        <div id="detail-{key}" class="detail-section">
            <h3>📊 相似面片 {i + 1} - 详细视图</h3>
            <div class="detail-content">
                <div class="detail-viewer">
                    {detail_htmls[key].split('<body>')[1].split('</body>')[0]}
                </div>
                <div class="detail-info">
                    {similar_info_html}
                </div>
            </div>
        </div>
                """)
        
        return ''.join(sections_html)
    
    def _generate_patch_info_panel(self, patch_info: Dict, is_query: bool = False) -> str:
        """生成面片信息面板HTML / Generate patch info panel HTML"""
        if not patch_info:
            return "<div class='info-group'><div class='info-title'>📊 信息不可用</div><p>未找到面片详细信息</p></div>"
        
        # 格式化函数 / Formatting functions
        def format_value(value, is_numeric=False):
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                if is_numeric:
                    return f'<span class="info-value numeric">{value:.4f}</span>' if isinstance(value, float) else f'<span class="info-value numeric">{value}</span>'
                else:
                    return f'{value:.4f}' if isinstance(value, float) else str(value)
            elif isinstance(value, str):
                return value[:50] + "..." if len(str(value)) > 50 else str(value)
            elif isinstance(value, list):
                return f"列表 ({len(value)} 项)"
            else:
                return str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        
        # 基本信息 / Basic Information
        basic_info = f"""
        <div class="info-group">
            <div class="info-title">📋 基本信息</div>
            <div class="info-item">
                <span class="info-label">面片ID</span>
                <span class="info-value">{format_value(patch_info.get('pattern_id'))}</span>
            </div>
            <div class="info-item">
                <span class="info-label">边数 (Sides)</span>
                <span class="info-value">{format_value(patch_info.get('sides'))}</span>
            </div>
            <div class="info-item">
                <span class="info-label">来源对象</span>
                <span class="info-value">{format_value(patch_info.get('source_obj'))}</span>
            </div>
            <div class="info-item">
                <span class="info-label">质量 (Quality)</span>
                <span class="info-value">{format_value(patch_info.get('quality'))}</span>
            </div>
        </div>"""
        
        # 几何信息 / Geometry Information
        geometry_info = f"""
        <div class="info-group">
            <div class="info-title">📐 几何信息</div>
            <div class="info-item">
                <span class="info-label">顶点数</span>
                <span class="info-value">{format_value(patch_info.get('num_vertices'))}</span>
            </div>
            <div class="info-item">
                <span class="info-label">面数</span>
                <span class="info-value">{format_value(patch_info.get('num_faces'))}</span>
            </div>
            <div class="info-item">
                <span class="info-label">面积</span>
                <span class="info-value">{format_value(patch_info.get('area'), True)}</span>
            </div>
            <div class="info-item">
                <span class="info-label">边界总长度</span>
                <span class="info-value">{format_value(patch_info.get('total_boundary_length'), True)}</span>
            </div>
            <div class="info-item">
                <span class="info-label">复杂度分数</span>
                <span class="info-value">{format_value(patch_info.get('complexity_score'), True)}</span>
            </div>
        </div>"""
        
        # 曲率信息 / Curvature Information
        curvature_info = f"""
        <div class="info-group">
            <div class="info-title">📈 曲率信息</div>
            <div class="info-item">
                <span class="info-label">平均曲率</span>
                <span class="info-value">{format_value(patch_info.get('avg_curvature'), True)}</span>
            </div>
            <div class="info-item">
                <span class="info-label">曲率方差</span>
                <span class="info-value">{format_value(patch_info.get('curvature_variance'), True)}</span>
            </div>
        </div>"""
        
        # 拓扑信息 / Topology Information
        topology_info = f"""
        <div class="info-group">
            <div class="info-title">🔗 拓扑信息</div>
            <div class="info-item">
                <span class="info-label">Canonical Form</span>
                <span class="info-value">{format_value(patch_info.get('canonical_form'))}</span>
            </div>
            <div class="info-item">
                <span class="info-label">EdgeBreaker 编码</span>
                <span class="info-value">{format_value(patch_info.get('edgebreaker_encoding'))}</span>
            </div>
        </div>"""
        
        # 几何数据统计 / Geometric Data Statistics
        geometry = patch_info.get('geometry', {})
        geometry_stats = ""
        if geometry:
            boundary_vertices = geometry.get('boundary_vertices', [])
            vertex_normals = geometry.get('vertex_normals', [])
            mean_curvatures = geometry.get('mean_curvatures', [])
            gaussian_curvatures = geometry.get('gaussian_curvatures', [])
            edge_lengths = geometry.get('edge_lengths', [])
            edge_curvatures = geometry.get('edge_curvatures', [])
            
            geometry_stats = f"""
        <div class="info-group">
            <div class="info-title">📊 几何数据统计</div>
            <div class="info-item">
                <span class="info-label">边界顶点</span>
                <span class="info-value">{len(boundary_vertices) if boundary_vertices else 0} 个</span>
            </div>
            <div class="info-item">
                <span class="info-label">顶点法向量</span>
                <span class="info-value">{len(vertex_normals) if vertex_normals else 0} 个</span>
            </div>
            <div class="info-item">
                <span class="info-label">平均曲率数据</span>
                <span class="info-value">{len(mean_curvatures) if mean_curvatures else 0} 个</span>
            </div>
            <div class="info-item">
                <span class="info-label">高斯曲率数据</span>
                <span class="info-value">{len(gaussian_curvatures) if gaussian_curvatures else 0} 个</span>
            </div>
            <div class="info-item">
                <span class="info-label">边长数据</span>
                <span class="info-value">{len(edge_lengths) if edge_lengths else 0} 个</span>
            </div>
            <div class="info-item">
                <span class="info-label">边曲率数据</span>
                <span class="info-value">{len(edge_curvatures) if edge_curvatures else 0} 个</span>
            </div>
        </div>"""
        
        return basic_info + geometry_info + curvature_info + topology_info + geometry_stats
    
    def _get_javascript_code(self) -> str:
        """获取JavaScript代码 / Get JavaScript code"""
        return """
        let activeDetails = [];
        const maxDetails = 2;

        function showDetail(detailId) {
            const element = document.getElementById(`detail-${detailId}`);
            const button = event.target;

            if (activeDetails.includes(detailId)) {
                // 如果已经显示，则隐藏 / If already shown, hide it
                element.style.display = 'none';
                button.classList.remove('active');
                activeDetails = activeDetails.filter(id => id !== detailId);
            } else {
                // 如果未显示，检查是否超过限制 / If not shown, check if exceeds limit
                if (activeDetails.length >= maxDetails) {
                    // 隐藏最早显示的 / Hide the earliest shown
                    const oldestId = activeDetails.shift();
                    document.getElementById(`detail-${oldestId}`).style.display = 'none';
                    document.querySelector(`button[onclick="showDetail('${oldestId}')"]`).classList.remove('active');
                }

                // 显示新的 / Show new one
                element.style.display = 'block';
                button.classList.add('active');
                activeDetails.push(detailId);

                // 滚动到视图 / Scroll to view
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }

        function hideAllDetails() {
            activeDetails.forEach(detailId => {
                document.getElementById(`detail-${detailId}`).style.display = 'none';
                document.querySelector(`button[onclick="showDetail('${detailId}')"]`).classList.remove('active');
            });
            activeDetails = [];
        }
        """