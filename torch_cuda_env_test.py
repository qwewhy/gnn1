import sys
import platform
import torch
import torch_geometric

def verify_environment():
    """
    一个全面的验证函数，用于检查GNN开发环境的每一个关键部分。
    """
    print("=" * 60)
    print(" GNN 环境与GPU加速功能全面验证报告")
    print("=" * 60)
    print(f"系统平台: {platform.system()} {platform.release()}")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"PyTorch Geometric 版本: {torch_geometric.__version__}")
    print("-" * 60)

    print("\n--- 1. PyTorch 与 CUDA 核心验证 ---")

    # 检查1：PyTorch是否能找到CUDA
    cuda_available = torch.cuda.is_available()
    print(f"1.1. PyTorch能否找到CUDA? -> {cuda_available}")
    if not cuda_available:
        print("\n!!! 致命错误：PyTorch无法检测到CUDA支持。!!!")
        print("请检查您的NVIDIA驱动是否正确安装，以及PyTorch是否安装了CUDA版本。")
        print("=" * 60)
        return

    # 检查2：PyTorch编译时所用的CUDA版本
    print(f"1.2. PyTorch编译时链接的CUDA版本: {torch.version.cuda}")
    print("     (这应与您安装时选择的版本[12.1]匹配，而非驱动版本)")

    # 检查3：检测到的GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"1.3. 检测到的GPU数量: {gpu_count}")

    if gpu_count > 0:
        # 检查4：主要GPU的名称
        gpu_name = torch.cuda.get_device_name(0)
        print(f"1.4. GPU 0 名称: {gpu_name}")
    else:
        print("\n!!! 警告：未检测到任何GPU设备。!!!")
        print("=" * 60)
        return

    print("\n--- 2. PyTorch Geometric 与 GPU 协同验证 ---")

    # 检查5：创建图数据并将其迁移至GPU
    try:
        # 定义一个简单图的边（一个三角形: 0-1, 1-2, 2-0）
        edge_index = torch.tensor([[0, 1, 2],
                                   [1, 2, 0]], dtype=torch.long)
        # 定义节点特征 (3个节点，每个节点16维特征)
        x = torch.randn(3, 16)

        # 创建一个PyG的Data对象
        data = torch_geometric.data.Data(x=x, edge_index=edge_index)
        print("2.1. 成功在CPU上创建PyG Data对象:")
        print(f"     {data}")

        # 关键测试：将图数据对象移动到GPU设备
        device = torch.device('cuda')
        data_on_gpu = data.to(device)
        print(f"\n2.2. 尝试将Data对象移动到设备: '{device}'")

        # 验证数据是否真的在GPU上
        node_features_on_cuda = data_on_gpu.x.is_cuda
        edge_indices_on_cuda = data_on_gpu.edge_index.is_cuda

        print(f"2.3. 验证数据位置:")
        print(f"     节点特征张量(x)是否在CUDA上? -> {node_features_on_cuda}")
        print(f"     边索引张量(edge_index)是否在CUDA上? -> {edge_indices_on_cuda}")

        if node_features_on_cuda and edge_indices_on_cuda:
            print("\n" + "=" * 60)
            print("✅ 恭喜！您的GNN开发环境已完全配置成功，并支持GPU加速！")
            print("=" * 60)
        else:
            raise RuntimeError("数据迁移至GPU后，张量位置验证失败。")

    except Exception as e:
        print(f"\n!!! 在PyG验证过程中发生错误: {e}!!!")
        print("这可能表明PyG的底层编译依赖库(torch-scatter, torch-sparse等)安装不正确。")
        print("请仔细回顾第三部分的安装步骤。")
        print("=" * 60)


if __name__ == "__main__":
    verify_environment()