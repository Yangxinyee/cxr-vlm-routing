#!/usr/bin/env python3
"""
Patch CheXagent modeling file for transformers compatibility.
- find_pruneable_heads_and_indices: removed from transformers.pytorch_utils
- get_head_mask: removed from PreTrainedModel
- dtype alignment: project vision features to language-model dtype
"""
from pathlib import Path

# Paths to patch: local model dir (when loading from disk) and HF cache
PATCH_PATHS = [
    Path("models/CheXagent-8b"),
    Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "CheXagent_hyphen_8b",
    Path.home() / ".cache" / "huggingface" / "hub" / "modules" / "transformers_modules" / "CheXagent_hyphen_8b",
]
MODELING_FILE = "modeling_chexagent.py"

OLD_IMPORT = "from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer"
NEW_CODE = '''from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer


def find_pruneable_heads_and_indices(
    heads, n_heads: int, head_size: int, already_pruned_heads: set
):
    """Compatibility shim: removed from transformers.pytorch_utils in newer versions."""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index'''

DTYPE_OLD = """        # step 4: get the embeddings of the prompt
        inputs_lang = self.language_model.get_input_embeddings()(input_ids)
        lang_atts = attention_mask"""
DTYPE_NEW = """        # step 4: get the embeddings of the prompt
        inputs_lang = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None and input_vis.dtype != inputs_lang.dtype:
            input_vis = input_vis.to(dtype=inputs_lang.dtype, device=inputs_lang.device)
        lang_atts = attention_mask"""


def main():
    patched = 0
    for base in PATCH_PATHS:
        path = base / MODELING_FILE
        if not path.exists():
            continue
        with open(path) as f:
            content = f.read()
        modified = False
        if OLD_IMPORT in content and "def find_pruneable_heads_and_indices" not in content:
            content = content.replace(OLD_IMPORT, NEW_CODE)
            modified = True
        # Patch 2: get_head_mask
        HEAD_MASK_ANCHOR = "        elif isinstance(module, nn.Linear) and module.bias is not None:\n            module.bias.data.zero_()\n\n\nclass CheXagentEncoder(nn.Module):"
        HEAD_MASK_PATCH = '''        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
        """Compatibility: get_head_mask removed from PreTrainedModel in newer transformers."""
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask


class CheXagentEncoder(nn.Module):'''
        if HEAD_MASK_ANCHOR in content and "def get_head_mask" not in content:
            content = content.replace(HEAD_MASK_ANCHOR, HEAD_MASK_PATCH)
            modified = True
        if DTYPE_OLD in content and DTYPE_NEW not in content:
            content = content.replace(DTYPE_OLD, DTYPE_NEW)
            modified = True
        if modified:
            with open(path, "w") as f:
                f.write(content)
            print(f"Patched: {path}")
            patched += 1
    if patched == 0:
        print("No files needed patching or CheXagent modeling file not found.")


if __name__ == "__main__":
    main()
