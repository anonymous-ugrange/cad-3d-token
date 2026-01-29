import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from Cad_VLM.config.macro import (
    N_BIT,
    END_PAD,
    BOOLEAN_PAD,
    ONE_EXT_SEQ_LENGTH,
    CAD_CLASS_INFO,
    eps,
)

N_EXT_TYPE = ONE_EXT_SEQ_LENGTH - 1
NUMERICAL_TOKEN_VAL = END_PAD + BOOLEAN_PAD


class CELoss(nn.Module):
    """
    Cross Entropy Loss for Text2CAD
    """

    def __init__(self, device):
        super(CELoss, self).__init__()

        self.ce_cad = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)
        self.ce_pc = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss()

    def forward(self, cad_dict: dict):
        """
        cad_dict: dictionary containing 'pred', 'target' and 'key_padding_mask' key.
                pred: shape (B,N,2)
                target: shape (B,N)
                key_padding_mask: shape (B,N)
        """

        key_padding_mask = cad_dict["key_padding_mask"]
        loss = []
        if cad_dict["key_padding_mask"] is not None:
            self.loss_seq_x = torch.sum(
                self.ce_cad(
                    cad_dict["pred"][:, :, 0].permute(0, 2, 1),
                    cad_dict["target"][:, :, 0].long(),
                )
                * key_padding_mask[:, :, 0]
            ) / torch.sum(key_padding_mask[:, :, 0] * 1)
            self.loss_seq_y = torch.sum(
                self.ce_cad(
                    cad_dict["pred"][:, :, 1].permute(0, 2, 1),
                    cad_dict["target"][:, :, 1].long(),
                )
                * key_padding_mask[:, :, 1]
            ) / torch.sum(key_padding_mask[:, :, 1] * 1)

        self.loss_seq = (self.loss_seq_x + self.loss_seq_y) / 2
        loss_keys = ["loss_seq"]

        result_dict = {
            key: getattr(self, key).detach().item() 
            for key in loss_keys
        }

        loss = self.loss_seq
        return loss, result_dict


class SpaceAwareLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = torch.device("cpu") if device is None else device

        self.mse = nn.MSELoss(reduction="none")
        self.softmax = nn.Softmax(dim=-1)

        pi = torch.pi
        self.register_buffer("range_sketch", torch.tensor([0.0, 1.0]))
        self.register_buffer(
            "range_extrusion",
            torch.tensor(
                [
                    [-1, 2],  # d+
                    [-1, 2],  # d-
                    [-1, 2],  # tx
                    [-1, 2],  # ty
                    [-1, 2],  # tz
                    [-pi, 2 * pi],  # rx
                    [-pi, 2 * pi],  # ry
                    [-pi, 2 * pi],  # rz
                    [0, 2],  # scale
                ]
            ),
        )
        self.augment = True

    def forward(self, pred, vec_dict, mask_cad_dict):
        # pred:[b,n,2,onehot]
        device = self.device

        batch = pred.shape[0]
        indices = torch.repeat_interleave(
            torch.arange(batch, device=device), mask_cad_dict["n_extrusion"]
        )

        save_dict = {}
        ext_idx_0, ext_idx_1 = self.get_ext_indices(mask_cad_dict["indices_extrusion"])

        target_ = vec_dict["cad_vec_target"][indices]  # (n_ext, n, 2)
        target_ext = target_[ext_idx_0, ext_idx_1, 0]  # (n_ext, 9)
        # topk around target
        save_dict.update({"skt_target": target_, "ext_target": target_ext})

        save_dict["target_skt_numerical"] = self.numericalize(
            target_, is_sketch=True, one_hot=False
        )
        save_dict["target_ext_numerical"] = self.numericalize(
            target_ext, is_sketch=False, one_hot=False
        )
        if self.augment:
            save_dict["augmented_target_ext_numerical"] = self.numericalize(
                self.batch_random_ext(target_ext),
                is_sketch=False,
                one_hot=False,
            )

        pred_ = pred[indices]  # (n_ext,n,2,onehot)
        pred_probs = self.softmax(pred_)  # (n_ext,n,2,onehot)
        pred_ext = pred_probs[ext_idx_0, ext_idx_1, 0]  # (n_ext, 9, onehot)
        save_dict["pred_skt_numerical"] = self.numericalize(
            pred_probs, save_dict=save_dict, is_sketch=True, one_hot=True
        )
        save_dict["pred_ext_numerical"] = self.numericalize(
            pred_ext, save_dict=save_dict, is_sketch=False, one_hot=True
        )

        skt_loss = self.sketch_loss(save_dict, mask_cad_dict)
        ext_loss = self.extrusion_loss(save_dict, mask_cad_dict)

        result_dict = {
            "sketch_loss": 100 * skt_loss.item(),
            "extrusion_loss": 100 * ext_loss.item(),
            "valid_sketch": save_dict["valid_sketch"].item(),
            "valid_extrusion": save_dict["valid_extrusion"].item(),
        }
        return skt_loss, ext_loss, result_dict

    def extrusion_loss(self, save_dict, mask_cad_dict):
        target_ext_repeat = torch.repeat_interleave(
            save_dict["target_ext_numerical"], repeats=N_EXT_TYPE, dim=0
        ).view(-1, N_EXT_TYPE, N_EXT_TYPE)
        pred_ext_repeat = target_ext_repeat.clone()
        pred_ext_repeat.diagonal(dim1=-2, dim2=-1).copy_(
            save_dict["pred_ext_numerical"]
        )

        if self.augment:
            augmented_target_ext_repeat = torch.repeat_interleave(
                save_dict["augmented_target_ext_numerical"], repeats=N_EXT_TYPE, dim=0
            ).view(-1, N_EXT_TYPE, N_EXT_TYPE)
            masku = torch.triu(
                torch.ones((N_EXT_TYPE, N_EXT_TYPE), device=pred_ext_repeat.device),
                diagonal=1,
            ).bool()
            pred_ext_repeat[:, masku] = augmented_target_ext_repeat[:, masku]
            target_ext_repeat[:, masku] = augmented_target_ext_repeat[:, masku]

        pred_ext_repeat = pred_ext_repeat.view(-1, N_EXT_TYPE)  # (9*n_ext, 9)
        target_ext_repeat = target_ext_repeat.view(-1, N_EXT_TYPE)

        target_skt_repeat = save_dict["target_skt_numerical"][:, None, ...]
        target_skt_repeat = target_skt_repeat.expand(-1, N_EXT_TYPE, -1, -1).flatten(0, 1)

        mask = mask_cad_dict["mask_sketch_points"][:, None, ...]
        mask = mask.expand(-1, N_EXT_TYPE, -1).flatten(0, 1)  # (9*n_ext, n)

        n_valid = torch.sum(mask * 2)
        save_dict["valid_extrusion"] = n_valid

        # (9*n_ext,n,2,3)
        pred_3D_repeat = SpaceAwareLoss.convert3D(
            points=target_skt_repeat, ext=pred_ext_repeat
        )

        target_3D_repeat = SpaceAwareLoss.convert3D(
            points=target_skt_repeat, ext=target_ext_repeat
        )

        ext_loss = self.mse(pred_3D_repeat, target_3D_repeat).mean(
            dim=-1
        )  # (9*n_ext,n,2)
        ext_loss = torch.sum(ext_loss * mask[:, :, None]) / n_valid
        return ext_loss

    def sketch_loss(self, save_dict, mask_cad_dict):
        mask = mask_cad_dict["mask_sketch_points"]  # (n_ext,n)
        n_valid = torch.sum(mask * 2)
        save_dict["valid_sketch"] = n_valid

        if self.augment:
            pred_3D = SpaceAwareLoss.convert3D(
                points=save_dict["pred_skt_numerical"],
                ext=save_dict["augmented_target_ext_numerical"],
            )
            target_3D = SpaceAwareLoss.convert3D(
                points=save_dict["target_skt_numerical"],
                ext=save_dict["augmented_target_ext_numerical"],
            )
        else:
            pred_3D = SpaceAwareLoss.convert3D(
                points=save_dict["pred_skt_numerical"],
                ext=save_dict["target_ext_numerical"],
            )
            target_3D = SpaceAwareLoss.convert3D(
                points=save_dict["target_skt_numerical"],
                ext=save_dict["target_ext_numerical"],
            )

        skt_loss = self.mse(pred_3D, target_3D).mean(dim=-1)  # (n_ext,n,2)
        skt_loss = torch.sum(skt_loss * mask[:, :, None]) / n_valid
        return skt_loss

    @staticmethod
    def convert3D(points, ext):
        # points:[n_ext,n,2], ext:[n_ext,9]
        n_ext, n = points.shape[:2]
        # d+,d-
        d_0 = ext[:, 0][:, None, None].broadcast_to(n_ext, n, 1)
        d_1 = -ext[:, 1][:, None, None].broadcast_to(n_ext, n, 1)
        pts_0 = torch.cat([points, d_0], dim=-1)
        pts_1 = torch.cat([points, d_1], dim=-1)
        # points: [n_ext,n,2,3]
        points = torch.cat([pts_0[:, :, None, :], pts_1[:, :, None, :]], dim=2)
        # scale
        points = points * ext[:, 8][:, None, None, None]
        # rotation
        R = SpaceAwareLoss.euler_to_rotation(ext[:, 5:8]).transpose(-1, -2)
        points = torch.einsum("bnkc,bcd->bnkd", points, R)
        # translation
        points = points + ext[:, 2:5][:, None, None, :]
        return points

    @staticmethod
    def euler_to_rotation(angles):
        theta = angles[:, 0]
        phi = angles[:, 1]
        gamma = angles[:, 2]

        cx, cy, cz = torch.cos(theta), torch.cos(phi), torch.cos(gamma)
        sx, sy, sz = torch.sin(theta), torch.sin(phi), torch.sin(gamma)

        R = torch.stack(
            [
                cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx,
                sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx,
                 -sy ,      cy*sx    ,     cy*cx     ,
            ],
            dim=-1,
        ).reshape(-1, 3, 3)
        return R

    def get_ext_indices(self, indices):
        # [d*2, t*3, r*3, boolean, scale]
        offsets = torch.arange(9)
        offsets[-1] = 9
        indices = indices[:, None] + offsets.to(self.device)
        return torch.arange(indices.shape[0], device=self.device)[:, None], indices

    def numericalize(self, x, save_dict=None, is_sketch=True, one_hot=True, n_bits=8):
        size = 2**n_bits

        if one_hot:
            # (n_ext,n,2,TOP_K) or (n_ext,9,TOP_K)
            indices = save_dict["skt_target"] if is_sketch else save_dict["ext_target"]
            indices = indices[..., None].long()
            probs = torch.gather(x, dim=-1, index=indices)
            if is_sketch:
                # (n_ext,n,2,TOP_K)
                range = self.range_sketch
                indices = range[0] + indices / (size - 1) * range[1]
            else:
                # (n_ext,9,TOP_K)
                range = self.range_extrusion
                indices = (
                    range[None, :, 0, None]
                    + indices / (size - 1) * range[None, :, 1, None]
                )
            x = torch.sum(indices * probs, dim=-1)
        else:
            if is_sketch:
                # x:[n_ext,n,2]
                range = self.range_sketch
                x = range[0] + (x - NUMERICAL_TOKEN_VAL) / (size - 1) * range[1]
            else:
                # x:[n_ext, 9]
                range = self.range_extrusion
                x = range[:, 0] + (x - NUMERICAL_TOKEN_VAL) / (size - 1) * range[:, 1]
        return x

    def batch_random_ext(self, target_ext, p=0.0):
        # target_ext:(n_ext, 9)
        n_ext = target_ext.shape[0]
        random_mask = torch.rand(n_ext) < p
        new_target_ext = torch.randint_like(
            target_ext,
            NUMERICAL_TOKEN_VAL,
            CAD_CLASS_INFO["one_hot_size"],
            dtype=target_ext.dtype,
            device=target_ext.device,
        )
        new_target_ext[~random_mask] = target_ext[~random_mask]
        return new_target_ext