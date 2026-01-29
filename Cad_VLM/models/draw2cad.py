import os, sys

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import torch.nn as nn
import torch
from Cad_VLM.config.macro import *

from Cad_VLM.models.layers.embedder import (
    CADSequenceEmbedder,
    PositionalEncodingSinCos,
    PositionalEncodingLUT,
)
from Cad_VLM.models.layers.attention import MultiHeadAttention
from Cad_VLM.models.layers.functional import FeedForwardLayer

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:4]))
from Cad_VLM.models.layers.improved_transformer import TransformerEncoderLayerImproved
from Cad_VLM.models.layers.transformer import TransformerEncoder
from Cad_VLM.models.draw_utils import _get_padding_mask_svg, _get_key_padding_mask_svg, _make_seq_first

from Cad_VLM.models.utils import count_parameters

from Cad_VLM.models.layers.utils_decode import (
    generate_attention_mask,
    create_flag_vec,
    create_index_vec,
    top_p_sampling,
)


class TokenController:
    def __init__(self, batch, device):
        self.counter = torch.zeros((batch, 1), dtype=torch.long, device=device)
    
    def update(self, new_token):
        #new_token: (b, 1, 2)
        self.counter -= 1
        new_token[(self.counter == 0).squeeze(-1), :, 0] = END_TOKEN.index("END_CURVE")
        self.counter[new_token[..., 0] == END_TOKEN.index("START_LINE")] = \
            OFFSET["line"] + 1
        self.counter[new_token[..., 0] == END_TOKEN.index("START_ARC")] = \
            OFFSET["arc"] + 1
        self.counter[new_token[..., 0] == END_TOKEN.index("START_CIRCLE")] = \
            OFFSET["circle"] + 1
        return new_token


class SVGEmbedding(nn.Module):
    """Embedding: view embed + command embed + parameter embed + positional embed"""
    def __init__(
        self, d_model, seq_len, input_option, svg_n_commands, 
        svg_n_args, args_dim, 
    ):
        super().__init__()

        """concatenation-based"""
        # 3x or 4x
        if input_option == "3x" or input_option == "4x":
            self.view_embed = nn.Embedding(4, 4)
            self.command_embed = nn.Embedding(svg_n_commands, 8)
        # 1x: keep dimension constant with other input option
        if input_option == "1x":
            self.command_embed = nn.Embedding(svg_n_commands, 12)

        args_dim = args_dim + 1
        self.args_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.args_mlp = nn.Linear(64 * svg_n_args, 128)
        self.mlp = nn.Linear(4 + 8 + 128, d_model)
        self.pos_encoding = PositionalEncodingLUT(
            d_model, max_len=seq_len + 2
        )

    def forward(self, view, command, args):
        assert command.shape == view.shape
        S, N = command.shape

        command_embedding = self.command_embed(command.long())
        args_embedding = self.args_mlp(self.args_embed((args + 1).long()).view(S, N, -1))

        """concatenation-based"""
        # 1x
        if S == 100:
            src = torch.cat([command_embedding, args_embedding], dim=-1)
        # 3x or 4x
        if S > 100:
            view_embedding = self.view_embed(view.long())
            src = torch.cat([view_embedding, command_embedding, args_embedding], dim=-1)
        
        src = self.mlp(src)
        src = self.pos_encoding(src)

        return src


class Encoder(nn.Module):
    def __init__(
        self, d_model, n_heads, n_layers, dim_feedforward, dropout, input_option, svg_max_total_len, svg_n_commands, svg_n_args, 
        args_dim
    ):
        super().__init__()

        view_num = int(input_option[0])
        seq_len = view_num * svg_max_total_len
        self.embedding = SVGEmbedding(
            d_model=d_model, seq_len=seq_len, input_option=input_option,
            svg_n_commands=svg_n_commands, svg_n_args=svg_n_args,
            args_dim=args_dim
        )

        encoder_layer = TransformerEncoderLayerImproved(d_model, n_heads, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)
    
    def forward(self, view, command, args):
        assert command.shape == view.shape
        padding_mask, key_padding_mask = _get_padding_mask_svg(command, seq_dim=0), _get_key_padding_mask_svg(command, seq_dim=0)
    
        src = self.embedding(view, command, args)

        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z.squeeze(0)


class Decoder(nn.Module):
    def __init__(
            self,
            cad_class_info,
            cdim,
            zdim,
            num_heads,
            d_latent,
            num_layers,
            dropout,
            device,
    ):
        super(Decoder, self).__init__()

        self.cad_embed = CADSequenceEmbedder(
            one_hot_size=cad_class_info["one_hot_size"],
            flag_size=cad_class_info["flag_size"],
            index_size=cad_class_info["index_size"],
            d_model=cdim,
            device=device,
        )

        self.pos_embed = PositionalEncodingSinCos(
            embedding_size=cdim, max_seq_len=MAX_CAD_SEQUENCE_LENGTH, device=device
        )

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [
                TransformerLayer(
                    cdim=cdim,
                    zdim=zdim,
                    num_heads=num_heads,
                    d_latent=d_latent,
                    dropout=dropout,
                    is_decoder=True,
                )
                for i in range(self.num_layers)
            ]
        )

        self.seq_output_x = nn.Linear(
            cdim, cad_class_info["one_hot_size"]
        )
        self.seq_output_y = nn.Linear(
            cdim, cad_class_info["one_hot_size"]
        )

        self.attention_scores = dict()

    def forward(self, Z, vec_dict, mask_cad_dict):
        num_seq = vec_dict["cad_vec"].shape[1]

        S = self.pos_embed(num_seq) + self.cad_embed(
            vec_dict, mask_cad_dict["key_padding_mask"]
        )  # (B,N1,cdim)
        
        for i in range(self.num_layers):
            S, self.attention_scores[f"block_level_{i}"] = \
                self.decoder_layers[i](
                    S, Z=Z, mask_cad_dict=mask_cad_dict,
                )

        output_x = self.seq_output_x(S)
        output_y = self.seq_output_y(S)

        S = torch.stack([output_x, output_y], dim=2)

        return S, {"attention_scores": self.attention_scores}

    def decode(
        self,
        Z,
        maxlen,
        nucleus_prob,
        topk_index,
        device,
    ):
        self.eval()
        batch = Z.shape[0]
        new_cad_seq_dict={
            "cad_vec": torch.tensor([[[1, 0]]]).repeat(batch, 1, 1).to(device),
            "flag_vec": torch.zeros(batch, 1).int().to(device),
            "index_vec": torch.zeros(batch, 1).int().to(device),
        }

        if not ORIGINAL:
            token_controller = TokenController(batch, device)
        
        # NOTE: Iteratively run the forward method till the end token is predicted.
        for t in range(1, maxlen):
            cad_pred, _ = self(
                Z,
                new_cad_seq_dict,
                {
                    "attn_mask": generate_attention_mask(
                        t, t, device=device
                    ),
                    "key_padding_mask": (new_cad_seq_dict["cad_vec"] == 0),
                }
            )

            # --------------------------------- Sampling --------------------------------- #
            # Hybrid-Sampling
            if nucleus_prob == 0:
                if t == 1:  # NOTE: Remove this part for top-1 sampling
                    new_token = torch.topk(cad_pred, topk_index, dim=-1).indices[
                        :, t - 1 : t, :, -1
                    ]
                else:
                    # NOTE: Keep this part only for top-1 sampling
                    new_token = torch.argmax(cad_pred, dim=-1)[:, t - 1 : t]
            # Nucleus Sampling
            else:
                new_token = torch.cat(
                    [
                        top_p_sampling(cad_pred[:, t - 1 : t, 0], nucleus_prob),
                        top_p_sampling(cad_pred[:, t - 1 : t, 1], nucleus_prob),
                    ],
                    axis=-1,
                )

            # ------------------------------ CAD Sequence Update ------------------------------ #
            if not ORIGINAL:
                new_token = token_controller.update(new_token)

            # Add the new token (no masking here)
            new_cad_seq_dict["cad_vec"] = torch.cat(
                [new_cad_seq_dict["cad_vec"], new_token], axis=1
            )

            # ------------------------------ Flag generation ----------------------------- #
            # Create flag seq (Very important. Wrong flag may result in invalid model)
            new_cad_seq_dict["flag_vec"] = torch.cat(
                [
                    new_cad_seq_dict["flag_vec"],
                    create_flag_vec(
                        new_cad_seq_dict["cad_vec"], new_cad_seq_dict["flag_vec"]
                    ),
                ],
                axis=1,
            )

            # ----------------------------- Index Generation ----------------------------- #
            # Create index seq  (Very important. Wrong index may result in invalid model)
            new_cad_seq_dict["index_vec"] = torch.cat(
                [
                    new_cad_seq_dict["index_vec"],
                    create_index_vec(
                        new_cad_seq_dict["cad_vec"], new_cad_seq_dict["index_vec"]
                    ),
                ],
                axis=1,
            )

            # ------------------------- Masking the dummy tokens ------------------------- #
            # Mask the dummy tokens in the new CAD tokens (Very important. Wrong masking may result in inaccurate model)

            end_tokens=torch.logical_or(new_cad_seq_dict['cad_vec'][:,:,0] <= END_TOKEN.index("END_EXTRUSION"),new_cad_seq_dict['flag_vec']>0)
    

            num_tokens=new_cad_seq_dict["cad_vec"][
                end_tokens
            ].shape[0]

            mask = torch.cat(
                [
                    torch.ones((num_tokens, 1), dtype=torch.int32),
                    torch.zeros((num_tokens, 1), dtype=torch.int32),
                ],
                axis=1,
            ).to(device)
            
            new_cad_seq_dict["cad_vec"][
                end_tokens
            ] *= mask

        return new_cad_seq_dict


class TransformerLayer(nn.Module):
    def __init__(
        self,
        cdim,
        num_heads,
        d_latent,
        dropout,
        is_decoder=False,
        zdim=None,
    ):
        super(TransformerLayer, self).__init__()

        # Multi-Head Self Attention for CAD Sequence
        # TODO: Check if Flash Attention is implemented
        #! Note: Dropout is set to 0 for attention otherwise the sum of attention weights > 1
        self.is_decoder = is_decoder

        self.sa_seq = MultiHeadAttention(
            input_dim=cdim,
            embed_dim=cdim,
            dropout=0,
            num_heads=num_heads,
        )

        if self.is_decoder:
            self.linear_decoder = nn.Linear(zdim, cdim)
            self.dropout_decoder = nn.Dropout(dropout)

        # LayerNormalization
        self.norm_seq = nn.ModuleDict(
            {
                "norm_1": nn.LayerNorm(cdim),
                "norm_2": nn.LayerNorm(cdim),
            }
        )

        # Dropout
        self.dp_seq = nn.ModuleDict(
            {
                "dropout_1": nn.Dropout(dropout),
                "dropout_2": nn.Dropout(dropout),
            }
        )

        # Feed forward Networks
        self.ffl_seq = FeedForwardLayer(input_dim=cdim, d_ff=d_latent)

        # Attention Scores
        self.attention_scores = dict()

    def forward(
        self,
        S,
        Z=None,
        mask_cad_dict=None,
    ):
        """
        S: tensor of shape (bs, num_seq, emb_dim)
        mask_cad_dict: dictionary with keys "attn_mask", "key_padding_mask"
        """

        self_attn_mask_dict = mask_cad_dict.copy()

        self_attn_mask_dict["key_padding_mask"] = torch.all(
            mask_cad_dict["key_padding_mask"], axis=2
        )

        #<----------  CAD SEQUENCE SELF-ATTENTION  ---------->
        S2 = self.norm_seq["norm_1"](S)  # (bs,num_seq,emb_dim)

        S2, S_score = self.sa_seq(
            S2,
            S2,
            S2,
            key_padding_mask=self_attn_mask_dict["key_padding_mask"],
            attn_mask=self_attn_mask_dict["attn_mask"],
        )  # (bs,num_seq,emb_dim) (Self-Attention)
        # (bs,num_seq,emb_dim) (Dropout + Addition)
        S = S + self.dp_seq["dropout_1"](S2)

        if self.is_decoder:
            Z = self.linear_decoder(Z)
            S = S + self.dropout_decoder(Z.unsqueeze(1))

        S2 = self.norm_seq["norm_2"](S)  # (bs,num_seq,emb_dim)

        # ? <---------- FEED-FORWARD + DROPOUT + ADDITION +    ---------->
        S = S + self.dp_seq["dropout_2"](self.ffl_seq(S2))

        # Add the cross attention scores (metadata)
        self.attention_scores["sa"] = S_score

        return S, self.attention_scores


class Bottleneck(nn.Module):
    def __init__(self, d_model):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(d_model, d_model // 2),
                                        nn.GELU(),
                                        nn.Linear(d_model // 2, d_model))

    def forward(self, z):
        return z + self.bottleneck(z)


class SVG2CADTransformer(nn.Module):
    def __init__(
        self, d_model, n_heads, n_layers, dim_feedforward, dropout,
        input_option, svg_max_total_len, svg_n_commands, svg_n_args,
        args_dim, device
    ):
        super(SVG2CADTransformer, self).__init__()

        self.encoder = Encoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            input_option=input_option, svg_max_total_len=svg_max_total_len,
            svg_n_commands=svg_n_commands, svg_n_args=svg_n_args,
            args_dim=args_dim
        )

        self.bottleneck = Bottleneck(d_model=d_model)

        self.decoder = Decoder(
            cad_class_info=CAD_CLASS_INFO, cdim=d_model, zdim=d_model,
            num_heads=n_heads, d_latent=dim_feedforward, 
            num_layers=n_layers, dropout=dropout, device=device
        )

    def forward(self, vec_dict, mask_cad_dict, svg_dict):
        views_enc_, commands_enc_, args_enc_ = _make_seq_first(
            svg_dict["view"], svg_dict["command"], svg_dict["args"]
        )  # Possibly None, None, None

        Z = self.encoder(views_enc_, commands_enc_, args_enc_)
        Z = self.bottleneck(Z)
        
        S, _ = self.decoder(Z, vec_dict, mask_cad_dict)
        return S, None
    
    def test_decode(
        self, svg_dict, maxlen, nucleus_prob, topk_index, device
    ):
        views_enc_, commands_enc_, args_enc_ = _make_seq_first(
            svg_dict["view"], svg_dict["command"], svg_dict["args"]
        )  # Possibly None, None, None

        Z = self.encoder(views_enc_, commands_enc_, args_enc_)
        Z = self.bottleneck(Z)

        S_output = self.decoder.decode(
            Z=Z, maxlen=maxlen, nucleus_prob=nucleus_prob,
            topk_index=topk_index, device=device
        )
        return S_output

    @staticmethod
    def from_config(cfg):
        return SVG2CADTransformer(
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"], input_option=cfg["input_option"],
            svg_max_total_len=SVG_MAX_TOTAL_LEN, 
            svg_n_commands=SVG_N_COMMANDS, svg_n_args=SVG_N_ARGS,
            args_dim=ARGS_DIM, device=cfg["device"]
        )

    def total_parameters(self, description=False, in_millions=False):
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000  # Convert to millions
            print(f"Number of Parameters: {num_params_million:.1f}M")
        else:
            num_params = count_parameters(self, description)
            print(f"Number of Parameters: {num_params}")

    def get_trainable_state_dict(self):
        # Get the state dict of the model which are trainable parameters
        return {
            k: v for k, v in self.state_dict().items() if "base_text_embedder" not in k.split(".")
        }