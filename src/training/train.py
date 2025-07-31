import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import yaml
from pathlib import Path
import tqdm
import numpy as np
import os

from src.models.gnn_encoder import DualInputGNNEncoder
from src.data_processing.pyg_dataset import PatchDataset
from src.data_processing.triplet_generator import TripletGenerator

def train(config_path: str):
    """
    GNN模型训练主循环 / Main training loop for GNN model.
    
    Args:
        config_path (str): YAML配置文件路径 / Path to YAML config file.
    """
    # 1. 加载配置 / Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 创建数据集和数据加载器 / Create Datasets and DataLoaders
    data_root = Path(config['data']['root'])
    patch_dataset = PatchDataset(root=str(data_root))
    
    # 创建三元组数据集，包装TripletGenerator / Create TripletDataset wrapping TripletGenerator
    class TripletDataset(torch.utils.data.Dataset):
        def __init__(self, mesh_path, patch_dataset, num_triplets):
            self.triplet_generator = TripletGenerator(mesh_path, patch_dataset)
            self.num_triplets = num_triplets
        
        def __len__(self):
            return self.num_triplets
            
        def __getitem__(self, idx):
            triplet = None
            while triplet is None:
                triplet = self.triplet_generator.generate_triplet()
            return triplet

    # 使用网格作为几何查询源 / Use meshes as source for geometric queries
    # Construct absolute paths from the project root
    project_root = Path(__file__).parent.parent.parent
    train_mesh_path = str(project_root / config['data']['train_mesh'])
    val_mesh_path = str(project_root / config['data']['val_mesh'])
    
    train_dataset = TripletDataset(train_mesh_path, patch_dataset, config['training']['num_triplets_train'])
    val_dataset = TripletDataset(val_mesh_path, patch_dataset, config['training']['num_triplets_val'])

    # 使用batch_size=1和梯度累积 / Use batch_size=1 with gradient accumulation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    pyg_loader = DataLoader(patch_dataset, batch_size=config['training']['batch_size'])


    # 3. 创建模型、损失函数和优化器 / Create Model, Loss, and Optimizer
    model = DualInputGNNEncoder(
        anchor_in_channels=config['model']['anchor_in_channels'],
        pattern_in_channels=config['model']['pattern_in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels'],
        gnn_type=config['model']['gnn_type']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=config['training']['margin'])

    # 4. 训练和验证循环 / Training and Validation Loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['training']['epochs']):
        # --- 训练阶段 / Training Phase ---
        model.train()
        total_train_loss = 0
        train_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        
        for anchor, positive, negative in train_bar:
            optimizer.zero_grad()
            
            # 手动移动数据到设备 / Move data to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # 模型前向传播 / Forward pass
            emb_a = model(anchor, input_type='anchor')
            emb_p = model(positive, input_type='pattern')
            emb_n = model(negative, input_type='pattern')

            loss = triplet_loss_fn(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # --- 验证阶段 / Validation Phase ---
        model.eval()
        total_val_loss = 0
        val_bar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
        
        with torch.no_grad():
            for anchor, positive, negative in val_bar:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                emb_a = model(anchor, input_type='anchor')
                emb_p = model(positive, input_type='pattern')
                emb_n = model(negative, input_type='pattern')
                
                loss = triplet_loss_fn(emb_a, emb_p, emb_n)
                total_val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 5. 模型检查点保存 / Model Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = checkpoint_dir / 'best_model.pt'
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} with val loss {avg_val_loss:.4f}")

    print("Training finished.")
    final_model_path = checkpoint_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == '__main__':
    # 命令行运行: python -m src.training.train / Command line: python -m src.training.train
    config_file = Path(__file__).parent.parent.parent / 'configs' / 'default_config.yaml'
    train(str(config_file))

