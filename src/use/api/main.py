# File: src/use/api/main.py
# FastAPI后端服务，替代原有的HTML生成方式

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
import traceback
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 导入你现有的模块
from src.use.visual_query_similar.config import ConfigManager
from src.use.visual_query_similar.database_manager import DatabaseManager
from src.use.visual_query_similar.mesh_loader import MeshLoader
from src.use.visual_query_similar.visualizer_core import VisualizerCore
from src.use.visual_query_similar.utils import Utils
from src.use.query_core import QueryEngine
from src.train.data_processing.pyg_dataset import PatchDataset
from src.train.data_processing.triplet_generator import TripletGenerator

def ensure_json_serializable(obj):
    """确保对象可以JSON序列化"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj

app = FastAPI(title="3D面片可视化查询API", version="1.0.0")

# 添加CORS中间件，允许React前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 临时允许所有域名来调试CORS问题
    allow_credentials=False,  # 不需要发送凭据
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
    expose_headers=["*"],
)

# 添加自定义中间件来确保CORS头被正确设置
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response

# 全局变量存储初始化的组件
config_manager = None
database_manager = None
mesh_loader = None
visualizer_core = None
query_engine = None
dataset = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化所有组件"""
    global config_manager, database_manager, mesh_loader, visualizer_core, query_engine, dataset
    
    try:
        print("🚀 正在初始化API服务...")
        
        # 设置环境
        Utils.setup_environment()
        
        # 检查依赖
        print("✅ 所有依赖项检查通过")
        
        # 查找配置文件
        config_path = Utils.find_config_file(project_root)
        if not config_path:
            raise Exception("无法找到配置文件")
        
        print(f"✅ 找到配置文件: {config_path}")
        
        # 初始化配置管理器
        print("📝 读取配置文件...")
        config_manager = ConfigManager(config_path)
        print(f"✅ 配置文件加载成功: {config_path}")
        
        # 初始化数据库管理器
        database_manager = DatabaseManager(config_manager.get_database_path())
        
        # 验证数据库结构
        columns = database_manager.get_database_columns()
        print(f"📋 数据库列名: {columns}")
        print("✅ 数据库结构检查通过")
        
        # 验证路径
        if not config_manager.validate_paths():
            raise Exception("必要文件路径验证失败")
        print("✅ 路径验证通过")
        
        # 初始化其他组件
        mesh_loader = MeshLoader()
        visualizer_core = VisualizerCore()
        
        # 初始化查询引擎
        print("🔧 初始化查询引擎...")
        query_engine = QueryEngine(config_path)
        
        # 加载数据集
        print("📊 加载数据集...")
        dataset = PatchDataset(root=str(config_manager.get_data_root()))
        
        print(f"✅ 可视化器初始化完成，数据库包含 {len(dataset)} 个面片")
        print("✅ API服务初始化成功")
        
        # 测试查询逻辑是否工作
        print("🧪 启动时测试查询逻辑...")
        test_mesh_files = mesh_loader.find_mesh_files(config_manager.get_project_root() / 'model')
        if test_mesh_files:
            test_mesh = test_mesh_files[0]
            try:
                generator = TripletGenerator(str(test_mesh), dataset)
                query_patch_indices = Utils.extract_random_patch_with_fallback(generator, mesh_index=0)
                if query_patch_indices:
                    query_anchor = Utils.create_anchor_with_fallback(generator, query_patch_indices, mesh_index=0)
                    if query_anchor is not None:
                        results = query_engine.query(query_anchor, k=3)
                        print(f"✅ 启动时查询测试成功: 找到 {len(results)} 个结果")
                    else:
                        print("⚠️ 启动时查询测试: 无法创建锚点")
                else:
                    print("⚠️ 启动时查询测试: 无法提取面片")
            except Exception as test_e:
                print(f"⚠️ 启动时查询测试失败: {test_e}")
        
    except Exception as e:
        print(f"❌ API服务初始化失败: {e}")
        traceback.print_exc()
        raise

# 请求/响应模型
class QueryRequest(BaseModel):
    mesh_path: str
    max_results: int = 6

class QueryResponse(BaseModel):
    success: bool
    message: str
    results: List[Dict[str, Any]] = []
    query_patch_info: Dict[str, Any] = {}

class MeshListResponse(BaseModel):
    meshes: List[Dict[str, str]]

class VisualizationResponse(BaseModel):
    success: bool
    patch_info: Dict[str, Any] = {}
    plotly_json: Dict[str, Any] = {}
    message: str = ""

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径，返回API信息"""
    return {
        "message": "3D面片可视化查询API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.options("/{path:path}")
async def options_handler(path: str):
    """处理所有OPTIONS请求（CORS预检）"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "false",
        }
    )

# 专门处理API路径的OPTIONS请求
@app.options("/api/{path:path}")
async def api_options_handler(path: str):
    """处理API路径的OPTIONS请求"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "false",
        }
    )

@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "dataset_size": len(dataset) if dataset else 0,
        "components_initialized": all([
            config_manager is not None,
            database_manager is not None,
            mesh_loader is not None,
            visualizer_core is not None,
            query_engine is not None,
            dataset is not None
        ])
    }

@app.get("/api/debug/test-query")
async def debug_test_query():
    """调试测试查询端点"""
    try:
        # 检查组件状态
        components_status = {
            "config_manager": config_manager is not None,
            "database_manager": database_manager is not None,
            "mesh_loader": mesh_loader is not None,
            "visualizer_core": visualizer_core is not None,
            "query_engine": query_engine is not None,
            "dataset": dataset is not None,
            "dataset_size": len(dataset) if dataset else 0
        }
        
        if not all([config_manager, database_manager, query_engine, dataset]):
            return {
                "success": False,
                "message": "组件未完全初始化",
                "components_status": components_status
            }
        
        # 获取一个测试网格文件
        model_dir = config_manager.get_project_root() / 'model'
        mesh_files = mesh_loader.find_mesh_files(model_dir)
        
        if not mesh_files:
            return {
                "success": False,
                "message": "没有找到网格文件",
                "components_status": components_status
            }
        
        test_mesh = mesh_files[0]
        
        # 尝试执行简化的查询流程
        try:
            generator = TripletGenerator(str(test_mesh), dataset)
            query_patch_indices = Utils.extract_random_patch_with_fallback(generator, mesh_index=0)
            
            if not query_patch_indices:
                return {
                    "success": False,
                    "message": "无法提取查询面片",
                    "components_status": components_status,
                    "test_mesh": str(test_mesh)
                }
            
            query_anchor = Utils.create_anchor_with_fallback(generator, query_patch_indices, mesh_index=0)
            
            if query_anchor is None:
                return {
                    "success": False,
                    "message": "无法创建查询锚点",
                    "components_status": components_status,
                    "test_mesh": str(test_mesh),
                    "patch_indices_count": len(query_patch_indices)
                }
            
            results = query_engine.query(query_anchor, k=3)
            
            return {
                "success": True,
                "message": f"测试查询成功，找到 {len(results)} 个结果",
                "components_status": components_status,
                "test_mesh": str(test_mesh),
                "patch_indices_count": len(query_patch_indices),
                "anchor_nodes": query_anchor.num_nodes,
                "results_count": len(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"查询测试失败: {str(e)}",
                "error_type": type(e).__name__,
                "components_status": components_status,
                "test_mesh": str(test_mesh) if 'test_mesh' in locals() else None
            }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"调试测试失败: {str(e)}",
            "error_type": type(e).__name__
        }

@app.get("/api/meshes", response_model=MeshListResponse)
async def get_available_meshes():
    """获取所有可用的网格文件"""
    try:
        model_dir = config_manager.get_project_root() / 'model'
        mesh_files = mesh_loader.find_mesh_files(model_dir)
        
        meshes = []
        for mesh_file in mesh_files:
            rel_path = mesh_file.relative_to(config_manager.get_project_root())
            meshes.append({
                "name": mesh_file.name,
                "path": str(mesh_file),
                "relative_path": str(rel_path)
            })
        
        return MeshListResponse(meshes=meshes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取网格文件失败: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def perform_query(request: QueryRequest):
    """执行面片相似性查询"""
    try:
        print(f"🎯 开始执行查询: {request.mesh_path}")
        print(f"🔧 组件状态检查...")
        print(f"   - config_manager: {'✅' if config_manager else '❌'}")
        print(f"   - database_manager: {'✅' if database_manager else '❌'}")
        print(f"   - query_engine: {'✅' if query_engine else '❌'}")
        print(f"   - dataset: {'✅' if dataset else '❌'} ({len(dataset) if dataset else 0} 个面片)")
        
        # 检查组件是否已初始化
        if not all([config_manager, database_manager, query_engine, dataset]):
            raise HTTPException(status_code=503, detail="服务尚未完全初始化，请稍后重试")
        
        # 1. 验证网格文件
        mesh_path = Path(request.mesh_path)
        if not mesh_path.exists():
            raise HTTPException(status_code=404, detail=f"网格文件不存在: {mesh_path}")
        
        print(f"✅ 网格文件存在: {mesh_path}")
        
        # 2. 提取查询面片
        print("🔍 提取查询面片...")
        generator = TripletGenerator(str(mesh_path), dataset)
        query_patch_indices = Utils.extract_random_patch_with_fallback(generator, mesh_index=0)
        if not query_patch_indices:
            raise HTTPException(status_code=500, detail="无法提取查询面片")
        
        print(f"✅ 提取到查询面片: {len(query_patch_indices)} 个面")
        
        # 3. 创建查询数据
        print("🔍 创建查询锚点...")
        query_anchor = Utils.create_anchor_with_fallback(generator, query_patch_indices, mesh_index=0)
        if query_anchor is None:
            raise HTTPException(status_code=500, detail="无法创建查询锚点")
        
        print(f"✅ 查询面片: {query_anchor.num_nodes} 个边界节点")
        
        # 4. 执行查询
        print("🔍 执行相似性查询...")
        results = query_engine.query(query_anchor, k=request.max_results)
        if not results:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "未找到相似结果",
                    "results": [],
                    "query_patch_info": {}
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Credentials": "false"
                }
            )
        
        print(f"✅ 查询完成，找到 {len(results)} 个相似面片")
        
        # 5. 格式化结果
        print("🔍 格式化查询结果...")
        formatted_results = []
        for idx, (db_index, similarity) in enumerate(results):
            patch_info = database_manager.get_patch_info_with_geometry(dataset, db_index)
            if not patch_info:
                patch_info = database_manager.get_patch_info(dataset, db_index)
                if patch_info:
                    patch_info['geometry'] = {}
            
            if patch_info:
                # 确保所有数据都是JSON可序列化的
                safe_patch_info = ensure_json_serializable(patch_info)
                formatted_results.append({
                    "rank": int(idx + 1),
                    "db_index": int(db_index),
                    "similarity": float(similarity),
                    "patch_info": safe_patch_info
                })
        
        # 6. 查询面片信息（确保JSON序列化兼容）
        query_patch_info = ensure_json_serializable({
            "patch_indices": query_patch_indices,
            "mesh_path": str(mesh_path),
            "mesh_name": mesh_path.name,
            "num_boundary_nodes": query_anchor.num_nodes
        })
        
        print(f"✅ 查询完成，找到 {len(formatted_results)} 个相似面片")
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"找到 {len(formatted_results)} 个相似面片",
                "results": formatted_results,
                "query_patch_info": query_patch_info
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "false"
            }
        )
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        traceback.print_exc()
        
        # 写入错误日志到文件
        try:
            with open("api_error.log", "a", encoding="utf-8") as f:
                f.write(f"\n=== Query Error at {__import__('datetime').datetime.now()} ===\n")
                f.write(f"Request: {request.mesh_path}, max_results: {request.max_results}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
        except:
            pass  # 忽略日志写入错误
        
        # 提供更详细的错误信息给前端
        error_detail = f"查询失败: {str(e)}"
        if "No such file or directory" in str(e):
            error_detail = f"网格文件路径错误: {request.mesh_path}"
        elif "TripletGenerator" in str(e):
            error_detail = f"面片提取失败: {str(e)}"
        elif "QueryEngine" in str(e):
            error_detail = f"查询引擎错误: {str(e)}"
            
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/api/visualize/{db_index}", response_model=VisualizationResponse)
async def get_patch_visualization(db_index: int):
    """获取特定面片的可视化数据"""
    try:
        print(f"🎨 生成面片可视化: DB索引 {db_index}")
        
        # 获取面片信息
        patch_info = database_manager.get_patch_info_with_geometry(dataset, db_index)
        if not patch_info:
            raise HTTPException(status_code=404, detail=f"面片不存在: DB索引 {db_index}")
        
        # 生成可视化数据
        fig = visualizer_core.create_single_patch_figure(patch_info)
        
        return VisualizationResponse(
            success=True,
            patch_info=patch_info,
            plotly_json=fig.to_dict(),
            message="可视化生成成功"
        )
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        traceback.print_exc()
        
        # 提供更详细的错误信息
        error_detail = f"可视化生成失败: {str(e)}"
        if "not found" in str(e).lower():
            error_detail = f"面片数据未找到 (DB索引: {db_index})"
        elif "database" in str(e).lower():
            error_detail = f"数据库查询失败: {str(e)}"
        elif "geometry" in str(e).lower():
            error_detail = f"几何数据处理失败: {str(e)}"
            
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/visualize/query")
async def get_query_patch_visualization(request: dict):
    """获取查询面片的可视化数据"""
    try:
        print(f"🎨 生成查询面片可视化")
        
        # 获取请求参数
        mesh_path = request.get('mesh_path')
        patch_indices = request.get('patch_indices', [])
        
        if not mesh_path:
            raise HTTPException(status_code=400, detail="缺少mesh_path参数")
        
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise HTTPException(status_code=404, detail=f"网格文件不存在: {mesh_path}")
        
        # 重新提取查询面片（为了获取几何数据）
        generator = TripletGenerator(str(mesh_path), dataset)
        
        # 如果提供了patch_indices，使用它们，否则重新提取
        if patch_indices:
            query_patch_indices = patch_indices
        else:
            query_patch_indices = Utils.extract_random_patch_with_fallback(generator, mesh_index=0)
            if not query_patch_indices:
                raise HTTPException(status_code=500, detail="无法提取查询面片")
        
        # 创建查询锚点
        query_anchor = Utils.create_anchor_with_fallback(generator, query_patch_indices, mesh_index=0)
        if query_anchor is None:
            raise HTTPException(status_code=500, detail="无法创建查询锚点")
        
        # 加载网格
        mesh = mesh_loader.load_mesh(str(mesh_path))
        if mesh is None:
            raise HTTPException(status_code=500, detail="无法加载网格文件")
        
        # 生成查询面片的可视化
        fig = visualizer_core.create_query_patch_figure(mesh, query_patch_indices, str(mesh_path))
        
        # 构建查询面片信息
        query_patch_info = {
            "pattern_id": "Query",
            "sides": len(query_patch_indices),
            "source_obj": mesh_path.name,
            "quality": "Query",
            "mesh_path": str(mesh_path),
            "num_faces": len(query_patch_indices),
            "num_vertices": len(set([v for face in [mesh.faces[i] for i in query_patch_indices] for v in face])),
            "num_boundary_nodes": query_anchor.num_nodes,
            "geometry": {}
        }
        
        return VisualizationResponse(
            success=True,
            patch_info=query_patch_info,
            plotly_json=fig.to_dict(),
            message="查询面片可视化生成成功"
        )
    except Exception as e:
        print(f"❌ 查询面片可视化生成失败: {e}")
        traceback.print_exc()
        
        error_detail = f"查询面片可视化生成失败: {str(e)}"
        if "not found" in str(e).lower() or "No such file" in str(e):
            error_detail = f"网格文件不存在: {mesh_path}"
        elif "TripletGenerator" in str(e):
            error_detail = f"面片提取失败: {str(e)}"
        elif "create_anchor" in str(e):
            error_detail = f"锚点创建失败: {str(e)}"
            
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/api/dataset/stats")
async def get_dataset_stats():
    """获取数据集统计信息"""
    try:
        stats = {
            "total_patches": len(dataset),
            "database_columns": database_manager.get_database_columns(),
            "data_root": str(config_manager.get_data_root()),
            "database_path": str(config_manager.get_database_path()),
            "project_root": str(config_manager.get_project_root())
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.get("/api/query/random")
async def get_random_query():
    """获取一个随机的查询样本（用于演示）"""
    try:
        import random
        
        # 随机选择一个数据集样本
        random_index = random.randint(0, len(dataset) - 1)
        random_patch = dataset[random_index]
        
        patch_info = database_manager.get_patch_info_with_geometry(dataset, random_index)
        if not patch_info:
            patch_info = database_manager.get_patch_info(dataset, random_index)
        
        return {
            "success": True,
            "db_index": random_index,
            "patch_info": patch_info,
            "message": f"随机选择了DB索引 {random_index} 的面片"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取随机查询失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动FastAPI服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


