# File: src/train/inference/visualize_attention.py
# 注意力可视化模块 / Attention visualization module

import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Optional
# from sklearn.manifold import TSNE
# import umap

from src.train.models import ImprovedGATEncoder
from src.train.data_processing.pyg_dataset import PatchDataset


class AttentionVisualizer:
    """
    GAT注意力权重和嵌入空间可视化工具
    Visualization tool for GAT attention weights and embedding space
    """
    
    def __init__(self, model: ImprovedGATEncoder, dataset: PatchDataset):
        self.model = model
        self.dataset = dataset
        self.device = next(model.parameters()).device
        
    def visualize_attention_weights(self, data_idx: int, layer: int = 1, 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        可视化特定样本的注意力权重
        
        Args:
            data_idx: 数据集中的索引
            layer: 要可视化的GAT层 (1, 2, 或 3)
            save_path: 保存图片的路径
        """
        # 获取数据
        data = self.dataset[data_idx].to(self.device)
        
        # 获取注意力权重
        self.model.eval()
        with torch.no_grad():
            edge_index, attention_weights = self.model.get_attention_weights(
                data, input_type='pattern', layer=layer
            )
        
        # 创建NetworkX图
        G = nx.Graph()
        num_nodes = data.num_nodes
        
        # 添加节点
        for i in range(num_nodes):
            # 节点属性
            node_attrs = {
                'valence': data.x[i, 0].item(),
                'is_boundary': data.x[i, 1].item(),
                'is_corner': data.x[i, 2].item(),
            }
            G.add_node(i, **node_attrs)
        
        # 添加边和注意力权重
        edge_index_np = edge_index.cpu().numpy()
        attention_weights_np = attention_weights.cpu().numpy().flatten() # Flatten attention weights
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index_np[0, i], edge_index_np[1, i]
            weight = attention_weights_np[i]
            G.add_edge(src, dst, weight=weight)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 子图1：网络结构和注意力权重
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 节点颜色基于度数
        node_colors = [G.nodes[i]['valence'] for i in G.nodes()]
        
        # 边宽度基于注意力权重
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # 归一化权重用于可视化
        if weights:
            min_weight = min(weights)
            max_weight = max(weights)
            if max_weight > min_weight:
                normalized_weights = [(w - min_weight) / (max_weight - min_weight + 1e-8) for w in weights]
                edge_widths = [1 + 5 * w for w in normalized_weights]  # 1-6的宽度范围
            else:
                edge_widths = [1] * len(edges)
        else:
            edge_widths = [1] * len(edges)
        
        # 绘制网络
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              cmap='viridis', node_size=500, ax=ax1)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax1)
        
        ax1.set_title(f'GAT Layer {layer} - Attention Weights\nPattern ID: {getattr(data, "pattern_id", "N/A")}, Sides: {getattr(data, "num_sides", "N/A")}')

        ax1.axis('off')
        
        # 子图2：注意力权重分布
        if weights:
            ax2.hist(weights, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(weights), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(weights):.3f}')
            ax2.axvline(np.median(weights), color='green', linestyle='--', 
                       label=f'Median: {np.median(weights):.3f}')
            ax2.set_xlabel('Attention Weight')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Attention Weight Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def visualize_embedding_space(self, num_samples: int = 1000, 
                                method: str = 'tsne',
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        可视化学习到的嵌入空间
        
        Args:
            num_samples: 要可视化的样本数
            method: 降维方法 ('tsne' 或 'umap')
            save_path: 保存路径
        """
        # Skip this visualization if libraries are not installed.
        try:
            from sklearn.manifold import TSNE
            import umap
        except ImportError:
            print("TSNE or UMAP not installed. Skipping embedding space visualization.")
            return

        # 收集嵌入向量
        embeddings = []
        labels = []
        sides = []
        complexities = []
        
        self.model.eval()
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[idx].to(self.device)
                embedding = self.model.encoder(data, input_type='pattern')
                
                embeddings.append(embedding.cpu().numpy())
                labels.append(getattr(data, 'quality', 0))
                sides.append(getattr(data, 'num_sides', 0))
                complexities.append(getattr(data, 'complexity_score', 0))

        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        sides = np.array(sides)
        complexities = np.array(complexities)
        
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        else:  # umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. 按质量标签着色
        scatter1 = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=labels, cmap='coolwarm', alpha=0.6, s=50)
        axes[0, 0].set_title('Embedding Space - Colored by Quality')
        axes[0, 0].set_xlabel('Component 1')
        axes[0, 0].set_ylabel('Component 2')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Quality (0=old, 1=new)')
        
        # 2. 按边数着色
        scatter2 = axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=sides, cmap='viridis', alpha=0.6, s=50)
        axes[0, 1].set_title('Embedding Space - Colored by Number of Sides')
        axes[0, 1].set_xlabel('Component 1')
        axes[0, 1].set_ylabel('Component 2')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Number of Sides')
        
        # 3. 按复杂度着色
        scatter3 = axes[1, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=complexities, cmap='plasma', alpha=0.6, s=50)
        axes[1, 0].set_title('Embedding Space - Colored by Complexity Score')
        axes[1, 0].set_xlabel('Component 1')
        axes[1, 0].set_ylabel('Component 2')
        plt.colorbar(scatter3, ax=axes[1, 0], label='Complexity Score')
        
        # 4. 密度图
        from scipy.stats import gaussian_kde
        xy = embeddings_2d.T
        z = gaussian_kde(xy)(xy)
        scatter4 = axes[1, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=z, cmap='hot', alpha=0.6, s=50)
        axes[1, 1].set_title('Embedding Space - Density')
        axes[1, 1].set_xlabel('Component 1')
        axes[1, 1].set_ylabel('Component 2')
        plt.colorbar(scatter4, ax=axes[1, 1], label='Density')
        
        plt.suptitle(f'Embedding Space Visualization ({method.upper()})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def analyze_attention_patterns(self, num_samples: int = 100) -> dict:
        """
        分析注意力模式的统计信息
        
        Returns:
            包含统计信息的字典
        """
        stats = {
            'attention_to_boundary': [],
            'attention_to_corner': [],
            'attention_to_high_valence': [],
            'attention_entropy': [],
            'max_attention_per_node': []
        }
        
        self.model.eval()
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[idx].to(self.device)
                
                # 获取所有层的注意力权重
                for layer in [1, 2, 3]:
                    edge_index, attention_weights = self.model.get_attention_weights(
                        data, input_type='pattern', layer=layer
                    )
                    
                    # 分析注意力分配
                    edge_index_np = edge_index.cpu().numpy()
                    attention_weights_np = attention_weights.cpu().numpy().flatten()
                    
                    # 计算每个节点接收的总注意力
                    node_attention = np.zeros(data.num_nodes)
                    for i in range(edge_index.shape[1]):
                        dst = edge_index_np[1, i]
                        node_attention[dst] += attention_weights_np[i]
                    
                    # 归一化
                    if node_attention.sum() > 0:
                        node_attention /= node_attention.sum()
                    
                    # 统计注意力分配给不同类型节点的比例
                    boundary_mask = data.x[:, 1].cpu().numpy() > 0.5
                    corner_mask = data.x[:, 2].cpu().numpy() > 0.5
                    high_valence_mask = data.x[:, 0].cpu().numpy() > 4
                    
                    stats['attention_to_boundary'].append(node_attention[boundary_mask].sum())
                    stats['attention_to_corner'].append(node_attention[corner_mask].sum())
                    stats['attention_to_high_valence'].append(node_attention[high_valence_mask].sum())
                    
                    # 计算注意力熵
                    attention_probs = attention_weights_np[attention_weights_np > 0]
                    if len(attention_probs) > 0:
                        entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-8))
                        stats['attention_entropy'].append(entropy)
                    
                    # 最大注意力值
                    if len(attention_weights_np) > 0:
                        stats['max_attention_per_node'].append(attention_weights_np.max())
        
        # 计算统计摘要
        summary = {}
        for key, values in stats.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary
    
    def create_attention_report(self, output_dir: str, num_samples: int = 10):
        """
        创建完整的注意力分析报告
        
        Args:
            output_dir: 输出目录
            num_samples: 要分析的样本数
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 分析注意力模式
        print("Analyzing attention patterns...")
        attention_stats = self.analyze_attention_patterns(num_samples=100)
        
        # 保存统计信息
        import json
        with open(output_path / 'attention_stats.json', 'w') as f:
            json.dump(attention_stats, f, indent=2)
        
        print("Attention pattern statistics:")
        for key, stats in attention_stats.items():
            print(f"\n{key}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}")
        
        # 2. 可视化个别样本
        print("\nVisualizing individual samples...")
        sample_indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            # 为每个GAT层创建可视化
            for layer in [1, 2, 3]:
                fig = self.visualize_attention_weights(
                    idx, layer=layer,
                    save_path=str(output_path / f'attention_sample{i}_layer{layer}.png')
                )
                plt.close(fig)
        
        # 3. 可视化嵌入空间
        print("\nVisualizing embedding space...")
        
        # t-SNE可视化
        fig_tsne = self.visualize_embedding_space(
            num_samples=1000, method='tsne',
            save_path=str(output_path / 'embedding_space_tsne.png')
        )
        if fig_tsne:
            plt.close(fig_tsne)
        
        # UMAP可视化
        fig_umap = self.visualize_embedding_space(
            num_samples=1000, method='umap',
            save_path=str(output_path / 'embedding_space_umap.png')
        )
        if fig_umap:
            plt.close(fig_umap)
        
        print(f"\nReport saved to {output_path}")


def main():
    """演示可视化功能"""
    import yaml
    
    # 加载配置
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载模型和数据集
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    encoder_config = {
        'anchor_in_channels': config['model']['anchor_in_channels'],
        'pattern_in_channels': config['model']['pattern_in_channels'],
        'hidden_channels': config['model']['hidden_channels'],
        'out_channels': config['model']['out_channels'],
        'num_heads': config['model']['num_heads'],
        'edge_dim': config['model']['edge_dim'],
        'dropout': 0  # 评估时不使用dropout
    }
    
    model = ImprovedGATEncoder(**encoder_config).to(device)
    
    # 加载训练好的权重
    checkpoint_path = Path(config['training']['checkpoint_dir']) / 'best_model.pt'
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No trained model found, using random weights")
    
    # 加载数据集
    dataset = PatchDataset(root=config['data']['root'])
    
    # 创建可视化器
    visualizer = AttentionVisualizer(model, dataset)
    
    # 生成报告
    visualizer.create_attention_report('visualization_results')


if __name__ == '__main__':
    main()
