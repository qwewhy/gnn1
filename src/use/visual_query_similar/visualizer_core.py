# File: src/use/visual_query_similar/visualizer_core.py
# 核心可视化模块 / Core visualization module

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    from .mesh_loader import MeshLoader
except ImportError:
    from mesh_loader import MeshLoader


class VisualizerCore:
    """核心可视化器 / Core Visualizer"""
    
    def __init__(self):
        """初始化可视化核心 / Initialize visualization core"""
        self.mesh_loader = MeshLoader()
    
    def create_overview_figure(self, mesh_path: str, query_patch_indices: list, 
                             results: List[Tuple[int, float]], max_results: int,
                             database_manager, dataset) -> go.Figure:
        """
        创建概览图 / Create overview figure
        
        Args:
            mesh_path: 网格文件路径 / Mesh file path
            query_patch_indices: 查询面片索引 / Query patch indices
            results: 查询结果 / Query results
            max_results: 最大结果数 / Maximum results
            database_manager: 数据库管理器 / Database manager
            dataset: 数据集 / Dataset
            
        Returns:
            plotly图表对象 / Plotly figure object
        """
        rows = 2
        cols = min(max_results // 2 + 1, 4)
        
        subplot_titles = ['🔍 查询面片 (点击放大)'] + [
            f'📊 相似面片 {i + 1} (点击放大)<br>相似度: {sim:.3f}'
            for i, (_, sim) in enumerate(results[:max_results - 1])
        ]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=subplot_titles
        )
        
        # 添加查询面片 / Add query patch
        mesh = self.mesh_loader.load_mesh(mesh_path)
        if mesh is not None:
            self._add_query_patch_to_subplot(fig, mesh, query_patch_indices, 1, 1)
        
        # 添加相似面片 / Add similar patches
        for idx, (db_index, similarity) in enumerate(results[:max_results - 1]):
            if idx >= (rows * cols - 1):
                break
                
            row = (idx + 1) // cols + 1
            col = (idx + 1) % cols + 1
            
            similar_patch_info = database_manager.get_patch_info_with_geometry(dataset, db_index)
            if similar_patch_info is None:
                similar_patch_info = database_manager.get_patch_info(dataset, db_index)
                if similar_patch_info:
                    similar_patch_info['geometry'] = {}
            
            if similar_patch_info:
                self._add_similar_patch_to_subplot(fig, similar_patch_info, row, col, similarity, mesh)
        
        # 增强布局设置 / Enhanced layout settings
        fig.update_layout(
            title=dict(
                text=f"🎯 面片相似性查询结果 - {Path(mesh_path).name}<br><sub>💡 点击任意面片可放大查看详细信息</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            height=800,
            showlegend=False,
            plot_bgcolor='rgba(240,240,240,0.3)',
        )
        
        # 为每个子图添加设置 / Add settings for each subplot
        for i in range(rows):
            for j in range(cols):
                scene_name = f'scene{i * cols + j + 1}' if i * cols + j > 0 else 'scene'
                if scene_name in fig.layout:
                    fig.layout[scene_name].update(
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                        bgcolor='rgba(255,255,255,0.8)',
                        xaxis=dict(showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=False, showticklabels=False),
                        zaxis=dict(showgrid=False, showticklabels=False)
                    )
        
        return fig
    
    def create_detail_views(self, mesh_path: str, query_patch_indices: list,
                          results: List[Tuple[int, float]], database_manager, dataset) -> Dict[str, go.Figure]:
        """
        创建详细视图 / Create detail views
        
        Args:
            mesh_path: 网格文件路径 / Mesh file path
            query_patch_indices: 查询面片索引 / Query patch indices
            results: 查询结果 / Query results
            database_manager: 数据库管理器 / Database manager
            dataset: 数据集 / Dataset
            
        Returns:
            详细视图字典 / Dictionary of detail views
        """
        detail_views = {}
        
        # 查询面片详细视图 / Query patch detail view
        query_detail_fig = go.Figure()
        mesh = self.mesh_loader.load_mesh(mesh_path)
        if mesh is not None:
            self._add_detailed_query_patch(query_detail_fig, mesh, query_patch_indices)
            query_detail_fig.update_layout(
                title="🔍 查询面片 - 详细视图",
                height=600,
                scene=dict(
                    camera=dict(eye=dict(x=2, y=2, z=2)),
                    bgcolor='rgba(255,255,255,1.0)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                    zaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                    aspectmode='data'
                )
            )
            detail_views['query'] = query_detail_fig
        
        # 相似面片详细视图 / Similar patches detail views
        for idx, (db_index, similarity) in enumerate(results[:5]):  # 前5个相似面片 / First 5 similar patches
            similar_patch_info = database_manager.get_patch_info_with_geometry(dataset, db_index)
            if similar_patch_info:
                detail_fig = go.Figure()
                self._add_detailed_similar_patch(detail_fig, similar_patch_info, similarity, mesh)
                detail_fig.update_layout(
                    title=f"📊 相似面片 {idx + 1} - 详细视图 (相似度: {similarity:.3f})",
                    height=600,
                    scene=dict(
                        camera=dict(eye=dict(x=2, y=2, z=2)),
                        bgcolor='rgba(255,255,255,1.0)',
                        aspectmode='data'
                    )
                )
                detail_views[f'similar_{idx}'] = detail_fig
        
        return detail_views
    
    def _add_query_patch_to_subplot(self, fig: go.Figure, mesh, patch_indices: list, row: int, col: int):
        """在子图中添加查询面片 / Add query patch to subplot"""
        # 整个网格（透明线框） / Entire mesh (transparent wireframe)
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            opacity=0.1,
            color='lightgray',
            showscale=False,
            name='网格主体'
        ), row=row, col=col)
        
        # 高亮面片（红色线框） / Highlight patch (red wireframe)
        if patch_indices:
            patch_faces = mesh.faces[patch_indices]
            
            # 绘制面片的线框 / Draw patch wireframe
            edges = []
            for face in patch_faces:
                edges.extend([
                    [face[0], face[1]], [face[1], face[2]], [face[2], face[0]]
                ])
            
            # 去重边 / Remove duplicate edges
            unique_edges = []
            edge_set = set()
            for edge in edges:
                edge_key = tuple(sorted(edge))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    unique_edges.append(edge)
            
            # 绘制边 / Draw edges
            for edge in unique_edges:
                v1, v2 = edge
                fig.add_trace(go.Scatter3d(
                    x=[mesh.vertices[v1, 0], mesh.vertices[v2, 0]],
                    y=[mesh.vertices[v1, 1], mesh.vertices[v2, 1]],
                    z=[mesh.vertices[v1, 2], mesh.vertices[v2, 2]],
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False,
                    name='查询面片'
                ), row=row, col=col)
    
    def _add_similar_patch_to_subplot(self, fig: go.Figure, patch_info: Dict[str, Any], 
                                    row: int, col: int, similarity: float, mesh):
        """在子图中添加相似面片 / Add similar patch to subplot"""
        try:
            geometry = patch_info.get('geometry', {})
            boundary_vertices = geometry.get('boundary_vertices', [])
            
            if boundary_vertices and len(boundary_vertices) > 2:
                vertices = np.array(boundary_vertices)
                
                # 添加三角化网格表面 / Add triangulated mesh surface
                self._add_triangulated_patch_to_subplot(fig, vertices, row, col)
                
                # 添加增强的内部网格线 / Add enhanced internal wireframe
                self._add_enhanced_patch_wireframe_to_subplot(fig, vertices, row, col)
                
                # 边界点（红色，与查询面片一致） / Boundary points (red, consistent with query patch)
                fig.add_trace(go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode='markers',
                    marker=dict(size=6, color='red', opacity=0.9),
                    showlegend=False,
                    name=f'边界点 (ID:{patch_info.get("pattern_id", "N/A")})'
                ), row=row, col=col)
                
                # 边界线（红色粗线） / Boundary lines (red thick lines)
                closed_vertices = np.vstack([vertices, vertices[0]])
                fig.add_trace(go.Scatter3d(
                    x=closed_vertices[:, 0],
                    y=closed_vertices[:, 1],
                    z=closed_vertices[:, 2],
                    mode='lines',
                    line=dict(width=4, color='red', dash='solid'),
                    showlegend=False,
                    name='边界'
                ), row=row, col=col)
                
                # 信息标签 / Information label
                center = np.mean(vertices, axis=0)
                fig.add_trace(go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2] + 0.05],
                    mode='text',
                    text=[f"ID: {patch_info.get('pattern_id', 'N/A')}<br>"
                          f"边数: {patch_info.get('num_sides', 'N/A')}<br>"
                          f"质量: {'新' if patch_info.get('quality') == 'new' else '旧'}"],
                    textposition='middle center',
                    textfont=dict(size=8, color='darkblue'),
                    showlegend=False
                ), row=row, col=col)
            
            else:
                # 程序化生成的面片 / Procedurally generated patch
                self._add_procedural_patch_with_enhanced_wireframe(fig, patch_info, row, col, similarity)
                
        except Exception as e:
            print(f"添加相似面片可视化失败: {e}")
            self._add_text_only_visualization(fig, patch_info, row, col, similarity)
    
    def _add_patch_wireframe(self, fig: go.Figure, vertices: np.ndarray, row: Optional[int] = None, col: Optional[int] = None):
        """为面片添加网格线 / Add wireframe to patch"""
        if len(vertices) < 3:
            return
            
        center = np.mean(vertices, axis=0)
        
        # 检查是否是子图网格 / Check if it's subplot grid
        is_subplot = row is not None and col is not None and hasattr(fig, '_grid_ref')
        
        # 从中心到每个顶点的线 / Lines from center to each vertex
        for i, vertex in enumerate(vertices):
            trace = go.Scatter3d(
                x=[center[0], vertex[0]],
                y=[center[1], vertex[1]],
                z=[center[2], vertex[2]],
                mode='lines',
                line=dict(color='lightblue', width=1, dash='dot'),
                showlegend=False,
                opacity=0.6
            )
            
            if is_subplot:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)
        
        # 顶点之间的连线 / Connections between vertices
        for i in range(len(vertices)):
            next_i = (i + 1) % len(vertices)
            trace = go.Scatter3d(
                x=[vertices[i, 0], vertices[next_i, 0]],
                y=[vertices[i, 1], vertices[next_i, 1]],
                z=[vertices[i, 2], vertices[next_i, 2]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False,
                opacity=0.7
            )
            
            if is_subplot:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)
    
    def _add_procedural_patch_with_enhanced_wireframe(self, fig: go.Figure, patch_info: Dict[str, Any], 
                                                    row: int, col: int, similarity: float):
        """根据拓扑信息生成增强的程序化几何表示 / Generate enhanced procedural geometry based on topology"""
        try:
            num_sides = patch_info.get('num_sides', 4)
            if isinstance(num_sides, str):
                num_sides = 4
            
            # 生成正多边形边界 / Generate regular polygon boundary
            angles = np.linspace(0, 2 * np.pi, num_sides + 1)
            radius = 1.0
            
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            z = np.zeros_like(x)
            
            vertices = np.column_stack([x[:-1], y[:-1], z[:-1]])
            
            # 添加三角化网格表面 / Add triangulated mesh surface
            self._add_triangulated_patch_to_subplot(fig, vertices, row, col)
            
            # 添加增强网格线 / Add enhanced wireframe
            self._add_enhanced_patch_wireframe_to_subplot(fig, vertices, row, col)
            
            # 边界点（红色） / Boundary points (red)
            fig.add_trace(go.Scatter3d(
                x=x[:-1], y=y[:-1], z=z[:-1],
                mode='markers',
                marker=dict(size=6, color='red', opacity=0.9),
                showlegend=False
            ), row=row, col=col)
            
            # 边界线（红色粗线） / Boundary lines (red thick lines)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(width=4, color='red'),
                showlegend=False
            ), row=row, col=col)
            
            # 信息标签 / Information label
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0.1],
                mode='text',
                text=[f"ID: {patch_info.get('pattern_id', 'N/A')}<br>"
                      f"边数: {num_sides}<br>"
                      f"质量: {'新' if patch_info.get('quality') == 'new' else '旧'}"],
                textposition='middle center',
                textfont=dict(size=8, color='darkblue'),
                showlegend=False
            ), row=row, col=col)
            
        except Exception as e:
            print(f"增强程序化可视化失败: {e}")
            self._add_text_only_visualization(fig, patch_info, row, col, similarity)
    
    def _add_detailed_query_patch(self, fig: go.Figure, mesh, patch_indices: list):
        """添加详细的查询面片视图 / Add detailed query patch view"""
        # 整个网格（透明） / Entire mesh (transparent)
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            opacity=0.15,
            color='lightgray',
            showscale=False,
            name='网格主体'
        ))
        
        # 高亮面片（红色线框） / Highlight patch (red wireframe)
        if patch_indices:
            patch_faces = mesh.faces[patch_indices]
            
            # 详细的线框绘制 / Detailed wireframe drawing
            for face in patch_faces:
                edges = [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]]
                for edge in edges:
                    v1, v2 = edge
                    fig.add_trace(go.Scatter3d(
                        x=[mesh.vertices[v1, 0], mesh.vertices[v2, 0]],
                        y=[mesh.vertices[v1, 1], mesh.vertices[v2, 1]],
                        z=[mesh.vertices[v1, 2], mesh.vertices[v2, 2]],
                        mode='lines',
                        line=dict(color='red', width=6),
                        showlegend=False
                    ))
            
            # 高亮顶点 / Highlight vertices
            patch_vertices = set()
            for face in patch_faces:
                patch_vertices.update(face)
            patch_vertices = list(patch_vertices)
            
            fig.add_trace(go.Scatter3d(
                x=mesh.vertices[patch_vertices, 0],
                y=mesh.vertices[patch_vertices, 1],
                z=mesh.vertices[patch_vertices, 2],
                mode='markers',
                marker=dict(size=1.5, color='red', opacity=0.9),
                name='面片顶点'
            ))
    
    def _add_detailed_similar_patch(self, fig: go.Figure, patch_info: Dict[str, Any], similarity: float, mesh):
        """添加详细的相似面片视图 / Add detailed similar patch view"""
        geometry = patch_info.get('geometry', {})
        boundary_vertices = geometry.get('boundary_vertices', [])
        
        if boundary_vertices and len(boundary_vertices) > 2:
            vertices = np.array(boundary_vertices)
            
            # 详细的网格线 / Detailed wireframe
            self._add_patch_wireframe(fig, vertices)
            
            # 边界点 / Boundary points
            fig.add_trace(go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers',
                marker=dict(size=8, color='blue'),
                name='边界顶点'
            ))
            
            # 边界线 / Boundary lines
            closed_vertices = np.vstack([vertices, vertices[0]])
            fig.add_trace(go.Scatter3d(
                x=closed_vertices[:, 0],
                y=closed_vertices[:, 1],
                z=closed_vertices[:, 2],
                mode='lines',
                line=dict(width=4, color='red'),
                name='边界'
            ))
    
    def _add_text_only_visualization(self, fig: go.Figure, patch_info: Dict[str, Any], 
                                   row: int, col: int, similarity: float):
        """纯文本显示（回退方案） / Text-only display (fallback)"""
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='text',
            text=[f"ID: {patch_info.get('pattern_id', 'N/A')}<br>"
                  f"边数: {patch_info.get('num_sides', 'N/A')}<br>"
                  f"质量: {'新' if patch_info.get('quality') == 'new' else '旧'}<br>"
                  f"相似度: {similarity:.3f}"],
            textposition='middle center',
            showlegend=False
        ), row=row, col=col)
    
    def create_single_patch_figure(self, patch_info: Dict[str, Any]) -> go.Figure:
        """
        为单个面片创建可视化图表 / Create visualization figure for a single patch
        
        Args:
            patch_info: 面片信息字典 / Patch information dictionary
            
        Returns:
            plotly图表对象 / Plotly figure object
        """
        fig = go.Figure()
        
        try:
            geometry = patch_info.get('geometry', {})
            boundary_vertices = geometry.get('boundary_vertices', [])
            
            if boundary_vertices and len(boundary_vertices) > 2:
                vertices = np.array(boundary_vertices)
                
                # 创建三角化内部网格 / Create triangulated internal mesh
                self._add_triangulated_patch_mesh(fig, vertices)
                
                # 添加详细的网格线 / Add detailed wireframe
                self._add_enhanced_patch_wireframe(fig, vertices)
                
                # 边界点（红色，与查询面片一致） / Boundary points (red, consistent with query patch)
                fig.add_trace(go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode='markers',
                    marker=dict(size=8, color='red', opacity=0.9),
                    name='边界顶点'
                ))
                
                # 边界线（红色粗线） / Boundary lines (red thick lines)
                closed_vertices = np.vstack([vertices, vertices[0]])
                fig.add_trace(go.Scatter3d(
                    x=closed_vertices[:, 0],
                    y=closed_vertices[:, 1],
                    z=closed_vertices[:, 2],
                    mode='lines',
                    line=dict(width=6, color='red'),
                    name='边界'
                ))
                
                # 中心点标记 / Center point marker
                center = np.mean(vertices, axis=0)
                fig.add_trace(go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2]],
                    mode='markers',
                    marker=dict(size=12, color='orange', symbol='diamond'),
                    name='中心点'
                ))
            
            else:
                # 程序化生成的面片 / Procedurally generated patch
                num_sides = patch_info.get('sides', patch_info.get('num_sides', 4))
                if isinstance(num_sides, str):
                    num_sides = 4
                
                # 生成正多边形边界 / Generate regular polygon boundary
                angles = np.linspace(0, 2 * np.pi, num_sides + 1)
                radius = 1.0
                
                x = radius * np.cos(angles)
                y = radius * np.sin(angles)
                z = np.zeros_like(x)
                
                vertices = np.column_stack([x[:-1], y[:-1], z[:-1]])
                
                # 创建三角化内部网格 / Create triangulated internal mesh
                self._add_triangulated_patch_mesh(fig, vertices)
                
                # 添加详细的网格线 / Add detailed wireframe
                self._add_enhanced_patch_wireframe(fig, vertices)
                
                # 边界点（红色） / Boundary points (red)
                fig.add_trace(go.Scatter3d(
                    x=x[:-1], y=y[:-1], z=z[:-1],
                    mode='markers',
                    marker=dict(size=8, color='red', opacity=0.9),
                    name='边界顶点'
                ))
                
                # 边界线（红色粗线） / Boundary lines (red thick lines)
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(width=6, color='red'),
                    name='边界'
                ))
                
                # 中心点 / Center point
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='markers',
                    marker=dict(size=12, color='orange', symbol='diamond'),
                    name='中心点'
                ))
            
            # 配置布局 / Configure layout
            fig.update_layout(
                title=dict(
                    text=f"面片详细视图 - ID: {patch_info.get('pattern_id', 'N/A')}",
                    x=0.5,
                    font=dict(size=16)
                ),
                scene=dict(
                    camera=dict(eye=dict(x=2, y=2, z=2)),
                    bgcolor='rgba(255,255,255,1.0)',
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(200,200,200,0.3)',
                        title='X轴'
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(200,200,200,0.3)',
                        title='Y轴'
                    ),
                    zaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(200,200,200,0.3)',
                        title='Z轴'
                    ),
                    aspectmode='data'
                ),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.2)',
                    borderwidth=1
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
        except Exception as e:
            print(f"⚠️ 单个面片可视化失败: {e}")
            # 回退到文本显示 / Fallback to text display
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='text',
                text=[f"ID: {patch_info.get('pattern_id', 'N/A')}<br>"
                      f"边数: {patch_info.get('sides', patch_info.get('num_sides', 'N/A'))}<br>"
                      f"质量: {'新' if patch_info.get('quality') == 1 else '旧'}<br>"
                      f"可视化数据不可用"],
                textposition='middle center',
                textfont=dict(size=14, color='red'),
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"面片信息 - ID: {patch_info.get('pattern_id', 'N/A')}",
                scene=dict(
                    camera=dict(eye=dict(x=2, y=2, z=2)),
                    bgcolor='rgba(255,255,255,1.0)'
                )
            )
        
        return fig
    
    def _add_triangulated_patch_mesh(self, fig: go.Figure, vertices: np.ndarray):
        """
        为面片添加三角化内部网格 / Add triangulated internal mesh for patch
        
        Args:
            fig: plotly图表对象 / Plotly figure object
            vertices: 边界顶点数组 / Boundary vertices array
        """
        try:
            if len(vertices) < 3:
                return
            
            # 计算中心点 / Calculate center point
            center = np.mean(vertices, axis=0)
            
            # 创建扇形三角化 / Create fan triangulation
            # 将所有顶点与中心点连接，形成三角形
            all_vertices = np.vstack([vertices, center])
            
            # 创建三角形面 / Create triangular faces
            triangles = []
            for i in range(len(vertices)):
                next_i = (i + 1) % len(vertices)
                # 三角形：边界顶点i, 边界顶点i+1, 中心点
                triangles.append([i, next_i, len(vertices)])
            
            triangles = np.array(triangles)
            
            # 添加三角化网格（半透明） / Add triangulated mesh (semi-transparent)
            fig.add_trace(go.Mesh3d(
                x=all_vertices[:, 0],
                y=all_vertices[:, 1], 
                z=all_vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=0.3,
                color='lightblue',
                showscale=False,
                name='面片表面',
                hoverinfo='skip'
            ))
            
        except Exception as e:
            print(f"⚠️ 三角化网格创建失败: {e}")
    
    def _add_enhanced_patch_wireframe(self, fig: go.Figure, vertices: np.ndarray):
        """
        为面片添加增强的网格线效果 / Add enhanced wireframe effect for patch
        
        Args:
            fig: plotly图表对象 / Plotly figure object
            vertices: 边界顶点数组 / Boundary vertices array
        """
        try:
            if len(vertices) < 3:
                return
            
            center = np.mean(vertices, axis=0)
            
            # 1. 从中心到每个边界顶点的线 / Lines from center to each boundary vertex
            for i, vertex in enumerate(vertices):
                fig.add_trace(go.Scatter3d(
                    x=[center[0], vertex[0]],
                    y=[center[1], vertex[1]],
                    z=[center[2], vertex[2]],
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dot'),
                    showlegend=False,
                    opacity=0.7,
                    name='内部连接线'
                ))
            
            # 2. 边界顶点之间的连线（加强版） / Enhanced connections between boundary vertices
            for i in range(len(vertices)):
                next_i = (i + 1) % len(vertices)
                fig.add_trace(go.Scatter3d(
                    x=[vertices[i, 0], vertices[next_i, 0]],
                    y=[vertices[i, 1], vertices[next_i, 1]],
                    z=[vertices[i, 2], vertices[next_i, 2]],
                    mode='lines',
                    line=dict(color='darkblue', width=3),
                    showlegend=False,
                    opacity=0.8,
                    name='边界连接线'
                ))
            
            # 3. 添加中心点 / Add center point
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.8),
                showlegend=False,
                name='内部中心点'
            ))
            
        except Exception as e:
            print(f"⚠️ 增强网格线创建失败: {e}")
    
    def _add_triangulated_patch_to_subplot(self, fig: go.Figure, vertices: np.ndarray, row: int, col: int):
        """
        在子图中为面片添加三角化内部网格 / Add triangulated internal mesh for patch in subplot
        
        Args:
            fig: plotly图表对象 / Plotly figure object
            vertices: 边界顶点数组 / Boundary vertices array
            row: 子图行号 / Subplot row number
            col: 子图列号 / Subplot column number
        """
        try:
            if len(vertices) < 3:
                return
            
            # 计算中心点 / Calculate center point
            center = np.mean(vertices, axis=0)
            
            # 创建扇形三角化 / Create fan triangulation
            all_vertices = np.vstack([vertices, center])
            
            # 创建三角形面 / Create triangular faces
            triangles = []
            for i in range(len(vertices)):
                next_i = (i + 1) % len(vertices)
                triangles.append([i, next_i, len(vertices)])
            
            triangles = np.array(triangles)
            
            # 添加三角化网格（半透明） / Add triangulated mesh (semi-transparent)
            fig.add_trace(go.Mesh3d(
                x=all_vertices[:, 0],
                y=all_vertices[:, 1], 
                z=all_vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=0.2,
                color='lightblue',
                showscale=False,
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)
            
        except Exception as e:
            print(f"⚠️ 子图三角化网格创建失败: {e}")
    
    def _add_enhanced_patch_wireframe_to_subplot(self, fig: go.Figure, vertices: np.ndarray, row: int, col: int):
        """
        在子图中为面片添加增强的网格线效果 / Add enhanced wireframe effect for patch in subplot
        
        Args:
            fig: plotly图表对象 / Plotly figure object
            vertices: 边界顶点数组 / Boundary vertices array
            row: 子图行号 / Subplot row number
            col: 子图列号 / Subplot column number
        """
        try:
            if len(vertices) < 3:
                return
            
            center = np.mean(vertices, axis=0)
            
            # 从中心到每个边界顶点的线 / Lines from center to each boundary vertex
            for i, vertex in enumerate(vertices):
                fig.add_trace(go.Scatter3d(
                    x=[center[0], vertex[0]],
                    y=[center[1], vertex[1]],
                    z=[center[2], vertex[2]],
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dot'),
                    showlegend=False,
                    opacity=0.6
                ), row=row, col=col)
            
            # 边界顶点之间的连线 / Connections between boundary vertices
            for i in range(len(vertices)):
                next_i = (i + 1) % len(vertices)
                fig.add_trace(go.Scatter3d(
                    x=[vertices[i, 0], vertices[next_i, 0]],
                    y=[vertices[i, 1], vertices[next_i, 1]],
                    z=[vertices[i, 2], vertices[next_i, 2]],
                    mode='lines',
                    line=dict(color='darkblue', width=2),
                    showlegend=False,
                    opacity=0.7
                ), row=row, col=col)
            
        except Exception as e:
            print(f"⚠️ 子图增强网格线创建失败: {e}")
    
    def create_query_patch_figure(self, mesh, patch_indices: list, mesh_path: str) -> go.Figure:
        """
        为查询面片创建可视化图表 / Create visualization figure for query patch
        
        Args:
            mesh: 网格对象 / Mesh object
            patch_indices: 面片索引 / Patch indices
            mesh_path: 网格文件路径 / Mesh file path
            
        Returns:
            plotly图表对象 / Plotly figure object
        """
        fig = go.Figure()
        
        try:
            # 整个网格（透明） / Entire mesh (transparent)
            fig.add_trace(go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                opacity=0.15,
                color='lightgray',
                showscale=False,
                name='网格主体'
            ))
            
            # 高亮查询面片 / Highlight query patch
            if patch_indices:
                patch_faces = mesh.faces[patch_indices]
                
                # 详细的线框绘制 / Detailed wireframe drawing
                for face in patch_faces:
                    edges = [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]]
                    for edge in edges:
                        v1, v2 = edge
                        fig.add_trace(go.Scatter3d(
                            x=[mesh.vertices[v1, 0], mesh.vertices[v2, 0]],
                            y=[mesh.vertices[v1, 1], mesh.vertices[v2, 1]],
                            z=[mesh.vertices[v1, 2], mesh.vertices[v2, 2]],
                            mode='lines',
                            line=dict(color='red', width=4),
                            showlegend=False,
                            name='查询面片边缘'
                        ))
                
                # 高亮顶点 / Highlight vertices
                patch_vertices = set()
                for face in patch_faces:
                    patch_vertices.update(face)
                patch_vertices = list(patch_vertices)
                
                fig.add_trace(go.Scatter3d(
                    x=mesh.vertices[patch_vertices, 0],
                    y=mesh.vertices[patch_vertices, 1],
                    z=mesh.vertices[patch_vertices, 2],
                    mode='markers',
                    marker=dict(size=1.5, color='red', opacity=0.9),
                    name='查询面片顶点'
                ))
                
                # 面片中心标记 / Patch center marker
                patch_center = np.mean(mesh.vertices[patch_vertices], axis=0)
                fig.add_trace(go.Scatter3d(
                    x=[patch_center[0]],
                    y=[patch_center[1]],
                    z=[patch_center[2]],
                    mode='markers+text',
                    marker=dict(size=6, color='orange', symbol='diamond'),
                    textposition='top center',
                    textfont=dict(size=9, color='red'),
                    name='查询面片中心'
                ))
            
            # 配置布局 / Configure layout
            fig.update_layout(
                title=dict(
                    text=f"查询面片 - 来自 {Path(mesh_path).name}",
                    x=0.5,
                    font=dict(size=16)
                ),
                scene=dict(
                    camera=dict(eye=dict(x=2, y=2, z=2)),
                    bgcolor='rgba(255,255,255,1.0)',
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(200,200,200,0.3)',
                        title='X轴'
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(200,200,200,0.3)',
                        title='Y轴'
                    ),
                    zaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(200,200,200,0.3)',
                        title='Z轴'
                    ),
                    aspectmode='data'
                ),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.2)',
                    borderwidth=1
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
        except Exception as e:
            print(f"⚠️ 查询面片可视化失败: {e}")
            # 回退到文本显示 / Fallback to text display
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='text',
                text=[f"查询面片可视化<br/>来源: {Path(mesh_path).name}<br/>面片数: {len(patch_indices) if patch_indices else 0}<br/>可视化数据不可用"],
                textposition='middle center',
                textfont=dict(size=14, color='red'),
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"查询面片 - {Path(mesh_path).name}",
                scene=dict(
                    camera=dict(eye=dict(x=2, y=2, z=2)),
                    bgcolor='rgba(255,255,255,1.0)'
                )
            )
        
        return fig