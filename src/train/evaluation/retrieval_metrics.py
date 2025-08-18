# File: src/train/evaluation/retrieval_metrics.py
# mAP和检索指标评估器 / mAP and retrieval metrics evaluator

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import time
from pathlib import Path

class RetrievalEvaluator:
    """
    检索性能评估器 - 专门计算mAP等检索指标
    Retrieval performance evaluator - specialized for computing mAP and other retrieval metrics
    """
    
    def __init__(self, k_values=[1, 5, 10, 20]):
        """
        初始化检索评估器
        
        Args:
            k_values: 计算precision@k和recall@k的k值列表
        """
        self.k_values = k_values
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model_retrieval(self, model, dataset, device, max_samples=None, 
                                batch_size=32, save_embeddings=False, output_dir=None):
        """
        评估模型的检索性能
        
        Args:
            model: 训练好的模型
            dataset: 测试数据集
            device: 计算设备
            max_samples: 最大评估样本数（None表示全部）
            batch_size: 批次大小，用于内存管理
            save_embeddings: 是否保存嵌入向量
            output_dir: 输出目录
            
        Returns:
            包含mAP等指标的字典
        """
        self.logger.info("🎯 开始检索性能评估...")
        start_time = time.time()
        
        model.eval()
        
        # 提取所有样本的嵌入和标签
        embeddings, labels, metadata = self._extract_embeddings_batched(
            model, dataset, device, max_samples, batch_size
        )
        
        # 计算检索指标
        results = self._compute_retrieval_metrics(embeddings, labels)
        
        # 额外分析
        results.update(self._analyze_retrieval_quality(embeddings, labels))
        
        # 分析不同类别的性能
        results.update(self._analyze_per_class_performance(embeddings, labels, metadata))
        
        # 保存嵌入向量（可选）
        if save_embeddings and output_dir:
            self._save_embeddings(embeddings, labels, metadata, output_dir)
        
        elapsed_time = time.time() - start_time
        results['evaluation_time'] = elapsed_time
        
        self.logger.info(f"✅ 检索评估完成，耗时 {elapsed_time:.2f}s")
        return results
    
    def _extract_embeddings_batched(self, model, dataset, device, max_samples=None, batch_size=32):
        """批量提取嵌入向量以节省内存"""
        embeddings = []
        labels = []
        metadata = []
        
        # 限制样本数量
        if max_samples is not None:
            indices = torch.randperm(len(dataset))[:max_samples]
        else:
            indices = range(len(dataset))
        
        total_samples = len(indices)
        self.logger.info(f"提取 {total_samples} 个样本的嵌入向量...")
        
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_indices = list(indices)[i:i+batch_size]
                batch_embeddings = []
                batch_labels = []
                batch_metadata = []
                
                for idx in batch_indices:
                    try:
                        data = dataset[idx]
                        
                        # 确保数据在正确设备上
                        if hasattr(data, 'to'):
                            data = data.to(device)
                        
                        # 获取嵌入
                        if hasattr(data, 'to'):
                            emb = model(data, input_type='pattern')
                            batch_embeddings.append(emb.cpu())
                        else:
                            # 跳过非PyG Data对象
                            self.logger.warning(f"跳过样本 {idx}: 不是PyG Data对象")
                            continue
                        
                        # 获取标签和元数据
                        quality = data.quality.item() if torch.is_tensor(data.quality) else data.quality
                        batch_labels.append(quality)
                        
                        # 收集元数据
                        meta = {
                            'pattern_id': getattr(data, 'pattern_id', idx),
                            'num_sides': getattr(data, 'num_sides', -1),
                            'complexity_score': getattr(data, 'complexity_score', 0.0),
                            'canonical_form': getattr(data, 'canonical_form', ''),
                            'has_geometry': getattr(data, 'has_geometry', False)
                        }
                        batch_metadata.append(meta)
                        
                    except Exception as e:
                        self.logger.warning(f"跳过样本 {idx}: {e}")
                        continue
                
                if batch_embeddings:
                    embeddings.extend(batch_embeddings)
                    labels.extend(batch_labels)
                    metadata.extend(batch_metadata)
                
                # 显示进度
                if (i // batch_size) % 10 == 0:
                    progress = min(100, (i + batch_size) / total_samples * 100)
                    self.logger.info(f"进度: {progress:.1f}%")
        
        if not embeddings:
            raise ValueError("未能提取任何嵌入向量")
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.tensor(labels)
        
        self.logger.info(f"成功提取 {len(embeddings)} 个样本的嵌入向量")
        return embeddings, labels, metadata
    
    def _compute_retrieval_metrics(self, embeddings, labels):
        """计算核心检索指标"""
        n_samples = len(embeddings)
        
        # 归一化嵌入向量（使用cosine similarity）
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        all_aps = []
        precision_at_k = defaultdict(list)
        recall_at_k = defaultdict(list)
        ndcg_at_k = defaultdict(list)
        
        self.logger.info("计算mAP和相关指标...")
        
        for i in range(n_samples):
            query_emb = embeddings_norm[i:i+1]
            query_label = labels[i].item()
            
            # 计算与所有其他样本的相似度（cosine similarity）
            similarities = torch.mm(query_emb, embeddings_norm.t()).squeeze()
            
            # 排序（相似度从高到低）
            _, sorted_indices = torch.sort(similarities, descending=True)
            
            # 跳过查询样本自己
            sorted_indices = sorted_indices[sorted_indices != i]
            if len(sorted_indices) == 0:
                continue
                
            sorted_labels = labels[sorted_indices]
            sorted_similarities = similarities[sorted_indices]
            
            # 计算相关性（相同质量标签认为相关）
            relevance = (sorted_labels == query_label).float()
            
            # 跳过没有相关样本的查询
            total_relevant = relevance.sum().item()
            if total_relevant == 0:
                continue
            
            # 计算Average Precision
            ap = self._compute_average_precision(relevance.numpy())
            all_aps.append(ap)
            
            # 计算precision@k、recall@k和NDCG@k
            for k in self.k_values:
                if k <= len(relevance):
                    relevant_at_k = relevance[:k].sum().item()
                    
                    # Precision@k
                    precision_at_k[k].append(relevant_at_k / k)
                    
                    # Recall@k
                    recall_at_k[k].append(relevant_at_k / total_relevant)
                    
                    # NDCG@k
                    ndcg_score = self._compute_ndcg_at_k(relevance[:k].numpy(), k)
                    ndcg_at_k[k].append(ndcg_score)
            
            # 显示进度
            if (i + 1) % 100 == 0:
                progress = (i + 1) / n_samples * 100
                self.logger.info(f"mAP计算进度: {progress:.1f}%")
        
        # 汇总结果
        results = {}
        results['mAP'] = np.mean(all_aps) if all_aps else 0.0
        results['num_queries'] = len(all_aps)
        results['total_samples'] = n_samples
        
        # 计算各种@k指标的均值
        for k in self.k_values:
            results[f'precision@{k}'] = np.mean(precision_at_k[k]) if precision_at_k[k] else 0.0
            results[f'recall@{k}'] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0.0
            results[f'ndcg@{k}'] = np.mean(ndcg_at_k[k]) if ndcg_at_k[k] else 0.0
            
            # 计算标准差
            results[f'precision@{k}_std'] = np.std(precision_at_k[k]) if precision_at_k[k] else 0.0
            results[f'recall@{k}_std'] = np.std(recall_at_k[k]) if recall_at_k[k] else 0.0
        
        return results
    
    def _compute_average_precision(self, relevance):
        """计算单个查询的Average Precision"""
        if relevance.sum() == 0:
            return 0.0
        
        precision_at_k = []
        num_relevant = 0
        
        for k, rel in enumerate(relevance, 1):
            if rel:
                num_relevant += 1
            precision_at_k.append(num_relevant / k)
        
        # AP = 相关位置处precision的平均值
        ap = sum(p * r for p, r in zip(precision_at_k, relevance)) / relevance.sum()
        return ap
    
    def _compute_ndcg_at_k(self, relevance, k):
        """计算NDCG@k"""
        if relevance.sum() == 0:
            return 0.0
        
        # DCG@k
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance[:k]))
        
        # IDCG@k (理想DCG)
        ideal_relevance = np.sort(relevance)[::-1]  # 降序排列
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k]))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _analyze_retrieval_quality(self, embeddings, labels):
        """分析检索质量的额外指标"""
        results = {}
        
        # 1. 嵌入空间的分离度分析
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # 计算类别内/外相似度
        intra_similarities = []
        inter_similarities = []
        
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            indices = torch.where(mask)[0]
            
            if len(indices) > 1:
                # 类别内相似度
                intra_sim = similarity_matrix[indices][:, indices]
                # 去除对角线（自己和自己）
                intra_sim = intra_sim[~torch.eye(len(indices), dtype=bool)]
                intra_similarities.extend(intra_sim.tolist())
                
                # 类别间相似度
                other_mask = ~mask
                if other_mask.sum() > 0:
                    inter_sim = similarity_matrix[indices][:, other_mask]
                    inter_similarities.extend(inter_sim.flatten().tolist())
        
        if intra_similarities and inter_similarities:
            results['intra_class_similarity'] = np.mean(intra_similarities)
            results['inter_class_similarity'] = np.mean(inter_similarities)
            results['similarity_gap'] = results['intra_class_similarity'] - results['inter_class_similarity']
            results['separation_ratio'] = results['intra_class_similarity'] / (results['inter_class_similarity'] + 1e-8)
        
        # 2. 类中心距离
        if len(unique_labels) >= 2:
            centers = []
            for label in unique_labels:
                mask = (labels == label)
                if mask.sum() > 0:
                    center = embeddings_norm[mask].mean(dim=0)
                    centers.append(center)
            
            if len(centers) >= 2:
                center_distances = []
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = F.pairwise_distance(centers[i].unsqueeze(0), centers[j].unsqueeze(0)).item()
                        center_distances.append(dist)
                
                results['mean_center_distance'] = np.mean(center_distances)
                results['min_center_distance'] = np.min(center_distances)
        
        # 3. 嵌入向量的分布统计
        results['embedding_mean_norm'] = embeddings_norm.norm(dim=1).mean().item()
        results['embedding_std_norm'] = embeddings_norm.norm(dim=1).std().item()
        
        # 4. 维度利用率
        embedding_variance = embeddings_norm.var(dim=0)
        results['effective_dimensions'] = (embedding_variance > 0.01).sum().item()
        results['dimension_utilization'] = results['effective_dimensions'] / embeddings_norm.size(1)
        
        return results
    
    def _analyze_per_class_performance(self, embeddings, labels, metadata):
        """分析每个类别的检索性能"""
        unique_labels = torch.unique(labels)
        per_class_results = {}
        
        for label in unique_labels.tolist():
            label_mask = (labels == label)
            class_name = 'new' if label == 1 else 'old'
            
            per_class_results[f'{class_name}_count'] = label_mask.sum().item()
            
            # 计算该类别作为查询时的平均性能
            class_indices = torch.where(label_mask)[0]
            if len(class_indices) > 1:  # 至少需要2个样本
                class_aps = []
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                
                for idx in class_indices[:min(50, len(class_indices))]:  # 限制计算量
                    query_emb = embeddings_norm[idx:idx+1]
                    similarities = torch.mm(query_emb, embeddings_norm.t()).squeeze()
                    _, sorted_indices = torch.sort(similarities, descending=True)
                    sorted_indices = sorted_indices[sorted_indices != idx]
                    
                    if len(sorted_indices) > 0:
                        sorted_labels = labels[sorted_indices]
                        relevance = (sorted_labels == label).float()
                        
                        if relevance.sum() > 0:
                            ap = self._compute_average_precision(relevance.numpy())
                            class_aps.append(ap)
                
                if class_aps:
                    per_class_results[f'{class_name}_mAP'] = np.mean(class_aps)
        
        return per_class_results
    
    def _save_embeddings(self, embeddings, labels, metadata, output_dir):
        """保存嵌入向量和相关信息"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存嵌入向量
        torch.save({
            'embeddings': embeddings,
            'labels': labels,
            'metadata': metadata
        }, output_path / 'embeddings.pt')
        
        self.logger.info(f"嵌入向量已保存到 {output_path / 'embeddings.pt'}")
    
    def print_detailed_results(self, results):
        """打印详细的评估结果"""
        print("\n" + "="*70)
        print("🎯 检索性能评估结果 (Retrieval Performance Evaluation)")
        print("="*70)
        
        # 主要指标
        print(f"📊 核心指标 (Core Metrics):")
        print(f"   mAP: {results['mAP']:.4f}")
        print(f"   有效查询数: {results['num_queries']}")
        print(f"   总样本数: {results['total_samples']}")
        
        # Precision@K
        print(f"\n📈 Precision@K:")
        for k in self.k_values:
            if f'precision@{k}' in results:
                mean_val = results[f'precision@{k}']
                std_val = results.get(f'precision@{k}_std', 0)
                print(f"   P@{k}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Recall@K  
        print(f"\n📉 Recall@K:")
        for k in self.k_values:
            if f'recall@{k}' in results:
                mean_val = results[f'recall@{k}']
                std_val = results.get(f'recall@{k}_std', 0)
                print(f"   R@{k}: {mean_val:.4f} ± {std_val:.4f}")
        
        # NDCG@K
        print(f"\n🏆 NDCG@K:")
        for k in self.k_values:
            if f'ndcg@{k}' in results:
                print(f"   NDCG@{k}: {results[f'ndcg@{k}']:.4f}")
        
        # 嵌入质量分析
        if 'similarity_gap' in results:
            print(f"\n🔍 嵌入质量分析 (Embedding Quality Analysis):")
            print(f"   类内相似度: {results['intra_class_similarity']:.4f}")
            print(f"   类间相似度: {results['inter_class_similarity']:.4f}")
            print(f"   分离度: {results['similarity_gap']:.4f}")
            print(f"   分离比例: {results.get('separation_ratio', 0):.4f}")
        
        if 'mean_center_distance' in results:
            print(f"   平均类中心距离: {results['mean_center_distance']:.4f}")
            print(f"   最小类中心距离: {results['min_center_distance']:.4f}")
        
        # 维度利用率
        if 'dimension_utilization' in results:
            print(f"\n🔧 嵌入空间分析:")
            print(f"   有效维度数: {results['effective_dimensions']}")
            print(f"   维度利用率: {results['dimension_utilization']:.4f}")
            print(f"   平均向量范数: {results['embedding_mean_norm']:.4f}")
        
        # 类别性能
        print(f"\n📋 类别性能 (Per-Class Performance):")
        for key in ['new', 'old']:
            if f'{key}_count' in results:
                count = results[f'{key}_count']
                map_score = results.get(f'{key}_mAP', 0)
                print(f"   {key.upper()}类: {count} 样本, mAP: {map_score:.4f}")
        
        # 性能评价
        print(f"\n💡 性能评价 (Performance Assessment):")
        map_score = results['mAP']
        if map_score > 0.8:
            print("   🎉 优秀：检索性能非常好")
            assessment = "excellent"
        elif map_score > 0.6:
            print("   ✅ 良好：检索性能良好")
            assessment = "good"
        elif map_score > 0.4:
            print("   ⚠️ 一般：检索性能有待提高")
            assessment = "fair"
        elif map_score > 0.2:
            print("   ❌ 较差：检索性能需要改进")
            assessment = "poor"
        else:
            print("   💀 很差：检索性能需要大幅改进")
            assessment = "very_poor"
        
        # 改进建议
        print(f"\n🚀 改进建议 (Improvement Suggestions):")
        if assessment in ['poor', 'very_poor']:
            print("   • 检查数据质量和标签准确性")
            print("   • 调整损失函数权重，加强度量学习")
            print("   • 考虑增加训练数据或数据增强")
            print("   • 尝试不同的模型架构或超参数")
        elif assessment == 'fair':
            print("   • 优化硬三元组挖掘策略")
            print("   • 调整学习率和训练epoch数")
            print("   • 考虑加入对比学习损失")
        elif assessment == 'good':
            print("   • 微调超参数获得更好性能")
            print("   • 尝试模型蒸馏或集成方法")
        else:
            print("   • 性能已经很好，专注于应用部署")
        
        if 'evaluation_time' in results:
            print(f"\n⏱️ 评估耗时: {results['evaluation_time']:.2f}s")
        
        print("="*70)
    
    def analyze_failure_cases(self, model, dataset, device, num_examples=10, k=5):
        """分析检索失败的案例"""
        self.logger.info("🔍 分析检索失败案例...")
        
        embeddings, labels, metadata = self._extract_embeddings_batched(
            model, dataset, device, max_samples=min(200, len(dataset))
        )
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        failure_cases = []
        success_cases = []
        
        for i in range(min(num_examples * 2, len(embeddings))):
            query_emb = embeddings_norm[i:i+1]
            query_label = labels[i].item()
            query_meta = metadata[i]
            
            # 计算相似度并排序
            similarities = torch.mm(query_emb, embeddings_norm.t()).squeeze()
            _, sorted_indices = torch.sort(similarities, descending=True)
            
            # 分析top-k结果
            top_k_indices = sorted_indices[1:k+1]  # 跳过自己
            top_k_labels = labels[top_k_indices]
            top_k_similarities = similarities[top_k_indices]
            top_k_metadata = [metadata[idx] for idx in top_k_indices.tolist()]
            
            # 计算准确率
            correct = (top_k_labels == query_label).sum().item()
            accuracy = correct / k
            
            case_info = {
                'query_idx': i,
                'query_label': 'new' if query_label == 1 else 'old',
                'query_metadata': query_meta,
                'top_k_labels': ['new' if l == 1 else 'old' for l in top_k_labels.tolist()],
                'top_k_similarities': top_k_similarities.tolist(),
                'top_k_metadata': top_k_metadata,
                'accuracy': accuracy,
                'precision_at_k': accuracy
            }
            
            if accuracy < 0.4:  # 失败案例
                failure_cases.append(case_info)
            elif accuracy > 0.8:  # 成功案例
                success_cases.append(case_info)
            
            if len(failure_cases) >= num_examples and len(success_cases) >= num_examples:
                break
        
        # 打印分析结果
        print(f"\n🔍 检索案例分析 (Top-{k} Analysis)")
        print("="*60)
        
        if failure_cases:
            print(f"\n❌ 失败案例 ({len(failure_cases)} 个):")
            for i, case in enumerate(failure_cases[:5]):
                print(f"\n案例 {i+1}:")
                print(f"  查询: {case['query_label']} (ID: {case['query_metadata']['pattern_id']})")
                print(f"  边数: {case['query_metadata']['num_sides']}")
                print(f"  Top-{k} 结果: {case['top_k_labels']}")
                print(f"  准确率: {case['accuracy']:.2f}")
                print(f"  相似度: {[f'{s:.3f}' for s in case['top_k_similarities']]}")
        
        if success_cases:
            print(f"\n✅ 成功案例 ({len(success_cases)} 个):")
            for i, case in enumerate(success_cases[:3]):
                print(f"\n案例 {i+1}:")
                print(f"  查询: {case['query_label']} (ID: {case['query_metadata']['pattern_id']})")
                print(f"  边数: {case['query_metadata']['num_sides']}")
                print(f"  Top-{k} 结果: {case['top_k_labels']}")
                print(f"  准确率: {case['accuracy']:.2f}")
        
        return failure_cases, success_cases


def demonstrate_map_evaluation():
    """演示mAP评估的使用"""
    print("🎯 mAP评估系统演示")
    
    # 模拟数据
    batch_size = 100
    embedding_dim = 128
    num_classes = 2
    
    # 创建有意义的模拟嵌入（类别间有区别）
    embeddings = []
    labels = []
    
    for class_id in range(num_classes):
        class_size = batch_size // num_classes
        # 每个类别的嵌入集中在不同区域
        class_center = torch.randn(embedding_dim) * 2
        class_embeddings = class_center + torch.randn(class_size, embedding_dim) * 0.5
        
        embeddings.append(class_embeddings)
        labels.extend([class_id] * class_size)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)
    
    # 创建评估器
    evaluator = RetrievalEvaluator([1, 5, 10])
    
    # 计算指标
    results = evaluator._compute_retrieval_metrics(embeddings, labels)
    results.update(evaluator._analyze_retrieval_quality(embeddings, labels))
    
    # 显示结果
    evaluator.print_detailed_results(results)


if __name__ == '__main__':
    demonstrate_map_evaluation()
