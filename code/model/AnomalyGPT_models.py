import torch
from torch import nn
import numpy as np
# from datas.dataset_3d import  *
from torch.nn import functional as F


class TextPromptAdapter(nn.Module):
    """Learnable residual adapter for ImageBind text embeddings.

    Applies a bottleneck MLP with a learnable scaling factor so that
    at initialisation the adapter is close to an identity function
    (scale starts small, up_proj is zero-initialised).
    """

    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 128, scale_init: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, embed_dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up_proj(self.act(self.down_proj(x)))


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)

    
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i].transpose(0,1)
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
    
class PromptLearner(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, dim_out)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,4096,9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output


# ============================================================
# Component: Multi-Scale Anomaly Fusion
# 用途：取代 openllama.py 中的 torch.mean，自適應地融合 4 層 anomaly map
# 狀態：尚未接入，需要在 openllama.py 中替換 torch.mean 才會生效
# ============================================================
class MultiScaleAnomalyFusion(nn.Module):
    """
    V2: Per-pixel Layer Selection Fusion

    改進重點（相比 V1）：
        V1 問題：全局固定權重 + spatial mask，每個 pixel 用一樣的層權重
        V2 改進：每個 pixel 獨立決定要多少淺層、多少深層資訊
        - scratch 區域可以偏重淺層（紋理/邊緣）
        - 語義異常區域可以偏重深層（語義理解）

    架構：
        gate network: 3 層 Conv2d，輸入 [B, n_layers, H, W]
        輸出 per-pixel, per-layer 的 softmax 權重 [B, n_layers, H, W]
        加權求和得到最終 anomaly map [B, 1, H, W]

    參數量：約 10K（仍可忽略）
    """

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.n_layers = n_layers
        hidden = n_layers * 4

        # Gate network: 從 stacked anomaly maps 學習 per-pixel layer weights
        self.gate = nn.Sequential(
            nn.Conv2d(n_layers, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_layers, kernel_size=1),
        )

    def forward(self, anomaly_maps: list) -> torch.Tensor:
        """
        Args:
            anomaly_maps: list of [B, 224, 224]，長度 = n_layers

        Returns:
            [B, 1, 224, 224]：融合後的 anomaly map
        """
        stacked = torch.stack(anomaly_maps, dim=1)        # [B, 4, 224, 224]
        attn = self.gate(stacked)                          # [B, 4, 224, 224]
        attn = F.softmax(attn, dim=1)                      # per-pixel softmax over layers
        fused = (stacked * attn).sum(dim=1, keepdim=True)  # [B, 1, 224, 224]
        return fused


# ============================================================
# Component: Enhanced PromptLearner (with Cross-Attention)
# 用途：取代原本的 PromptLearner，加入 image-conditioned cross-attention
# 狀態：尚未接入，需要在 openllama.py 中替換 self.prompt_learner 才會生效
# ============================================================
class EnhancedPromptLearner(nn.Module):
    """
    改進版 PromptLearner，在原本的 CNN 基礎上加入 cross-attention 機制。

    原本的問題：
        1. base_prompts 是固定的隨機參數，跟當前圖片完全無關
        2. CNN 只看到 anomaly heatmap (1 channel)，不知道原始圖片是什麼物件
        3. 同樣的 anomaly map pattern，bottle 的裂痕和 screw 的刮痕意義完全不同，
           但原版 PromptLearner 無法區分

    改進方式：
        Path A (保留): CNN 從 anomaly map 提取空間特徵 → anomaly-aware prompts
        Path B (新增): Learnable queries 透過 cross-attention 與 image embedding 交互
                       → image-aware prompts（知道「這是什麼東西」）
        Gated Fusion:  用 sigmoid gate 自適應地融合兩路 prompts
                       gate 值接近 1 → 更依賴 anomaly 資訊
                       gate 值接近 0 → 更依賴 image 資訊

    輸出格式：
        跟原版一樣是 [B, 18, 4096]（18 個 prompt tokens），
        可以直接替換原版 PromptLearner，不需要改 prompt_wrap 或其他下游 code。

    參數量：約 1.3 億（主要來自 cross-attention 和 FFN 的 4096 維度）
    """

    def __init__(self, dim_in: int = 1, dim_out: int = 4096,
                 n_prompts: int = 9, n_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            dim_in:     anomaly map 的 channel 數，預設 1
            dim_out:    LLM embedding 維度，Vicuna-7B 是 4096
            n_prompts:  每路生成的 prompt 數量，預設 9（跟原版一致）
            n_heads:    cross-attention 的 head 數
            dropout:    attention 和 FFN 的 dropout rate
        """
        super().__init__()
        self.n_prompts = n_prompts
        self.dim_out = dim_out

        # === Path A: 原版 CNN（完全保留，處理 anomaly map）===
        # 輸入: [B, 1, 224, 224] → 輸出: [B, 4096, 3, 3]
        # reshape 後變成 [B, 9, 4096]
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 112x112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 56x56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 28x28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 14x14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 7x7

            nn.Conv2d(dim_in * 1024, dim_out, kernel_size=5, padding=0),
            # 輸出: [B, 4096, 3, 3] → reshape 成 [B, 9, 4096]
        )

        # === Path B: Cross-Attention（新增）===

        # 可學習的 prompt queries，類似 DETR 的 object queries
        # 這些 queries 會去「問」image embedding：這張圖片是什麼？哪裡有問題？
        self.prompt_queries = nn.Parameter(
            torch.randn(n_prompts, dim_out) * 0.02  # 小的初始值，穩定訓練
        )

        # Cross-Attention: queries attend to image features + anomaly prompts
        # Q = prompt_queries [B, 9, 4096]
        # K, V = image_embed + anomaly_prompts [B, 10, 4096]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_out,
            num_heads=n_heads,
            batch_first=True,    # 輸入格式為 [B, seq_len, dim]
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim_out)

        # FFN: cross-attention 後的 feed-forward network
        # 標準 transformer block 結構：Attention → LayerNorm → FFN → LayerNorm
        self.ffn = nn.Sequential(
            nn.Linear(dim_out, dim_out * 4),   # 4096 → 16384（標準 4x expansion）
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_out * 4, dim_out),   # 16384 → 4096
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim_out)

        # === Gated Fusion ===
        # 用 sigmoid gate 決定每個 prompt token 更依賴 anomaly 還是 image 資訊
        # 輸入: anomaly_prompts 和 image_aware_prompts 拼接 [B, 9, 8192]
        # 輸出: gate 值 [B, 9, 4096]，每個值在 [0, 1] 之間
        self.gate = nn.Sequential(
            nn.Linear(dim_out * 2, dim_out),
            nn.Sigmoid()
        )

    def forward(self, anomaly_map: torch.Tensor,
                image_embed: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            anomaly_map:  [B, 1, 224, 224] — 融合後的異常熱力圖
            image_embed:  [B, 1, 4096] — 來自 llama_proj 的 image embedding（可選）
                          如果不傳，行為退化成跟原版 PromptLearner 類似

        Returns:
            [B, 18, 4096] — 18 個 prompt tokens，跟原版格式完全相同
                            前 9 個 = gated fusion prompts
                            後 9 個 = anomaly-aware prompts (CNN 路徑)
        """
        B = anomaly_map.shape[0]

        # --- Path A: Anomaly-aware prompts (原版 CNN 路徑) ---
        anomaly_prompts = self.meta_net(anomaly_map)                    # [B, 4096, 3, 3]
        anomaly_prompts = anomaly_prompts.reshape(B, self.dim_out, self.n_prompts)  # [B, 4096, 9]
        anomaly_prompts = anomaly_prompts.transpose(-2, -1)             # [B, 9, 4096]

        # --- Path B: Image-aware prompts (新增 cross-attention 路徑) ---
        # 展開 queries 到 batch 維度
        queries = self.prompt_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 9, 4096]

        if image_embed is not None:
            # 組合 key/value：image embedding + anomaly prompts
            # 這樣 cross-attention 可以同時看到「原始圖片是什麼」和「哪裡有異常」
            kv = torch.cat([image_embed, anomaly_prompts], dim=1)  # [B, 10, 4096]

            # Cross-Attention + residual + LayerNorm
            attn_out, _ = self.cross_attn(queries, kv, kv)  # [B, 9, 4096]
            queries = self.norm1(queries + attn_out)

            # FFN + residual + LayerNorm
            ffn_out = self.ffn(queries)
            image_aware_prompts = self.norm2(queries + ffn_out)  # [B, 9, 4096]
        else:
            # 沒有 image_embed 時，直接用 queries 本身
            # （退化成類似原版的 base_prompts）
            image_aware_prompts = queries

        # --- Gated Fusion ---
        # 拼接兩路 prompts，用 gate 決定混合比例
        gate_input = torch.cat([anomaly_prompts, image_aware_prompts], dim=-1)  # [B, 9, 8192]
        g = self.gate(gate_input)  # [B, 9, 4096]，值在 [0, 1]

        # g 接近 1 → 更依賴 anomaly_prompts（CNN 看到的異常資訊）
        # g 接近 0 → 更依賴 image_aware_prompts（cross-attention 看到的圖片語義）
        fused_prompts = g * anomaly_prompts + (1 - g) * image_aware_prompts  # [B, 9, 4096]

        # 最終輸出：fused prompts + anomaly prompts = 18 個 tokens
        # 跟原版 PromptLearner 的輸出格式完全一致
        output = torch.cat([fused_prompts, anomaly_prompts], dim=1)  # [B, 18, 4096]

        return output