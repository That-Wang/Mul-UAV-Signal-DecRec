import torch


def inject_stride(weight_path, new_stride=32.0):
    # 加载模型权重
    ckpt = torch.load(weight_path)
    model = ckpt['model']

    # 注入 stride
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([new_stride])  # 根据实际值设置

    # 重新保存
    torch.save({'model': model}, weight_path)
    print(f"成功注入 stride={new_stride} 到 {weight_path}")


# 使用示例（假设你的模型实际 stride 是 16）
inject_stride("runs/train/yololite(5MP)_decrec_lsnr_allDA2/weights/best.pt", new_stride=32.0)