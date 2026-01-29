import torch

import sys, os
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))
from Cad_VLM.config.macro import SVG_EOS_IDX


def _make_seq_first(*args):
    # N, S, ... -> S, N, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)
    

def _get_key_padding_mask_svg(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    with torch.no_grad():
        # key_padding_mask = (commands == SVG_EOS_IDX).cumsum(dim=seq_dim) > 0
        key_padding_mask = commands == SVG_EOS_IDX

        if seq_dim == 0:
            return key_padding_mask.transpose(0, 1)
        return key_padding_mask

    
def _get_padding_mask_svg(commands, seq_dim=0, extended=False):
    with torch.no_grad():
        padding_mask = (commands == SVG_EOS_IDX).cumsum(dim=seq_dim) == 0
        padding_mask = padding_mask.float()

        if extended:
            # padding_mask doesn't include the final EOS, extend by 1 position to include it in the loss
            S = commands.size(seq_dim)
            torch.narrow(padding_mask, seq_dim, 3, S-3).add_(torch.narrow(padding_mask, seq_dim, 0, S-3)).clamp_(max=1)

        if seq_dim == 0:
            return padding_mask.unsqueeze(-1)
        return padding_mask