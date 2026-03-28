import torch

class PixelSmileConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_target": ("CONDITIONING", {"tooltip": "连接目标表情的 CLIPTextEncode 输出 (例如: happy)"}),
                "conditioning_neutral": ("CONDITIONING", {"tooltip": "连接中性表情的 CLIPTextEncode 输出 (例如: neutral)"}),
                "score": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "表情强度权重"}),
                "method": (["score_one_all", "score_one (last 7 tokens)"], {"default": "score_one_all"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("CONDITIONING",)
    FUNCTION = "apply_pixelsmile"
    CATEGORY = "conditioning/pixelsmile"
    DESCRIPTION = "对两个 Conditioning 进行 PixelSmile 算法的张量插值，用于精确控制表情强度。"

    def apply_pixelsmile(self, conditioning_target, conditioning_neutral, score, method):
        out = []
        
        # ComfyUI 的 CONDITIONING 是一个列表，里面是 [张量, kwargs字典]
        for i in range(len(conditioning_target)):
            tgt_tensor, tgt_kwargs = conditioning_target[i]
            
            # 尝试获取对应的中性条件，如果列表长度不一致则取最后一个
            neu_idx = min(i, len(conditioning_neutral) - 1)
            neu_tensor, neu_kwargs = conditioning_neutral[neu_idx]
            
            # 1. 动态对齐序列长度 (Qwen 编码的文本长度可能不一致)
            # 维度通常是: [batch_size, sequence_length, embedding_dim]
            seq_tgt = tgt_tensor.shape[1]
            seq_neu = neu_tensor.shape[1]
            max_seq = max(seq_tgt, seq_neu)
            
            # 如果序列长度不一致，在末尾用 0 填充 (Padding)
            if seq_tgt < max_seq:
                tgt_tensor = torch.nn.functional.pad(tgt_tensor, (0, 0, 0, max_seq - seq_tgt))
            if seq_neu < max_seq:
                neu_tensor = torch.nn.functional.pad(neu_tensor, (0, 0, 0, max_seq - seq_neu))
            
            # 2. 执行 PixelSmile 张量插值数学逻辑
            if method == "score_one_all":
                # 公式: V_neu + s * (V_tgt - V_neu)
                delta = tgt_tensor - neu_tensor
                result_tensor = neu_tensor + score * delta
                
            elif method == "score_one (last 7 tokens)":
                # 对应你提出的 suffix 切片逻辑
                if max_seq > 7:
                    prefix = tgt_tensor[:, :-7, :]
                    suffix_tgt = tgt_tensor[:, -7:, :]
                    suffix_neu = neu_tensor[:, -7:, :]
                    
                    delta = suffix_tgt - suffix_neu
                    suffix = suffix_neu + score * delta
                    result_tensor = torch.cat([prefix, suffix], dim=1)
                else:
                    # 如果文本过短，退化为全局插值
                    delta = tgt_tensor - neu_tensor
                    result_tensor = neu_tensor + score * delta

            # 3. 处理 ComfyUI 额外的条件参数 (如 pooled_output, attention_mask)
            result_kwargs = tgt_kwargs.copy()
            
            # 同步插值 pooled_output (如果模型使用了它)
            if "pooled_output" in result_kwargs and "pooled_output" in neu_kwargs:
                tgt_pooled = result_kwargs["pooled_output"]
                neu_pooled = neu_kwargs["pooled_output"]
                if tgt_pooled is not None and neu_pooled is not None:
                    delta_pooled = tgt_pooled - neu_pooled
                    result_kwargs["pooled_output"] = neu_pooled + score * delta_pooled
                    
            # 修复 attention_mask 的长度对齐
            if "attention_mask" in result_kwargs:
                attn_mask = result_kwargs["attention_mask"]
                if attn_mask is not None and attn_mask.shape[-1] < max_seq:
                    pad_amount = max_seq - attn_mask.shape[-1]
                    result_kwargs["attention_mask"] = torch.nn.functional.pad(attn_mask, (0, pad_amount), value=0)

            out.append([result_tensor, result_kwargs])
            
        return (out,)

# 注册节点到 ComfyUI
NODE_CLASS_MAPPINGS = {
    "PixelSmileConditioning": PixelSmileConditioning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelSmileConditioning": "PixelSmile Conditioning Interpolation"
}