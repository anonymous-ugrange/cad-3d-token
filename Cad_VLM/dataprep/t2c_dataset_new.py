import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
import json
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from config.macro import *
import numpy as np


class StartEnd:
    @staticmethod
    def convert_vec(cad_vec):
        #(n, 2)
        device = cad_vec.device
        dtype = cad_vec.dtype

        def generate_mask(n):
            return torch.full((n,), False, device=device)
        
        def get_primitive_pos(cad_vec):
            end_curve_tokens = cad_vec[:, 0] == END_TOKEN.index("END_CURVE")
            coordinate_tokens = (cad_vec[:, 0] >= END_PAD + BOOLEAN_PAD - len(CURVE_TYPE)) \
                & (cad_vec[:, 1] >= END_PAD + BOOLEAN_PAD - len(CURVE_TYPE))

            line_tokens = torch.cat([
                generate_mask(2),
                end_curve_tokens[2:] & coordinate_tokens[1:-1] & ~coordinate_tokens[:-2],
            ], dim=0)
            line_insert_pos = torch.where(line_tokens)[0]
        
            arc_tokens = torch.cat([
                generate_mask(3),
                end_curve_tokens[3:-1] & coordinate_tokens[2:-2] & coordinate_tokens[1:-3] \
                    & (coordinate_tokens[4:] | end_curve_tokens[:-4]),
                generate_mask(1),
            ], dim=0)
            arc_insert_pos = torch.where(arc_tokens)[0]

            circle_tokens = end_curve_tokens ^ line_tokens ^ arc_tokens
            circle_insert_pos = torch.where(circle_tokens)[0]

            return {
                "line": line_insert_pos,
                "arc": arc_insert_pos,
                "circle": circle_insert_pos,
            }
        
        cad_vec = cad_vec[:torch.where(cad_vec[..., 0] == END_TOKEN.index("START"))[0][1]+1]
        end_pos_dict = get_primitive_pos(cad_vec)
        cad_vec[cad_vec >= END_TOKEN.index("START_LINE")] += len(CURVE_TYPE)
        cad_vec[end_pos_dict["line"], 0] = END_TOKEN.index("START_LINE")
        cad_vec[end_pos_dict["arc"], 0] = END_TOKEN.index("START_ARC")
        cad_vec[end_pos_dict["circle"], 0] = END_TOKEN.index("START_CIRCLE")    

        insert_pos = torch.cat([
            end_pos_dict["line"] - OFFSET["line"],
            end_pos_dict["arc"] - OFFSET["arc"],
            end_pos_dict["circle"] - OFFSET["circle"],
        ], dim=0)
        assert len(insert_pos) == len(set(insert_pos.tolist())), "insert pos duplicate!"

        offsets = torch.zeros(len(cad_vec), dtype=torch.long, device=device)
        offsets[insert_pos] = 1
        offsets = torch.cumsum(offsets, dim=0)

        new_cad_vec = torch.zeros((len(cad_vec) + len(insert_pos), 2), 
                                dtype=dtype, device=device)
        new_cad_vec[torch.arange(len(cad_vec)) + offsets] = cad_vec

        line_pos = torch.where(new_cad_vec[..., 0] == END_TOKEN.index("START_LINE"))[0]
        new_cad_vec[line_pos, 0] = END_TOKEN.index("END_CURVE")
        new_cad_vec[line_pos - OFFSET["line"] - 1, 0] = END_TOKEN.index("START_LINE")

        line_pos = torch.where(new_cad_vec[..., 0] == END_TOKEN.index("START_ARC"))[0]
        new_cad_vec[line_pos, 0] = END_TOKEN.index("END_CURVE")
        new_cad_vec[line_pos - OFFSET["arc"] - 1, 0] = END_TOKEN.index("START_ARC")

        line_pos = torch.where(new_cad_vec[..., 0] == END_TOKEN.index("START_CIRCLE"))[0]
        new_cad_vec[line_pos, 0] = END_TOKEN.index("END_CURVE")
        new_cad_vec[line_pos - OFFSET["circle"] - 1, 0] = END_TOKEN.index("START_CIRCLE")

        return new_cad_vec

    @staticmethod
    def restore_vec(cad_vec):
        #(n, 2)
        if not ORIGINAL:
            start_token = (cad_vec[..., 0] == END_TOKEN.index("START_LINE")) | \
                (cad_vec[..., 0] == END_TOKEN.index("START_ARC")) | \
                (cad_vec[..., 0] == END_TOKEN.index("START_CIRCLE"))
            cad_vec = cad_vec[~start_token]
            target_token = cad_vec >= END_TOKEN.index("END_EXTRUSION")
            cad_vec[target_token] -= len(CURVE_TYPE)
        return cad_vec

    @staticmethod
    def generate_attention_mask(sz1, sz2=None, device='cpu', mask_start_token=True):
        if sz2 is None:
            sz2 = sz1
        mask = torch.triu(torch.full((sz1, sz2), float(
            '-inf'), device=device), diagonal=1)

        # Add Mask for start token
        if mask_start_token:
            mask[1:, 0] = float("-inf")
        return mask

    @staticmethod
    def generate_key_padding_mask(cad_vec):
        return cad_vec == END_TOKEN.index("PADDING")
    
    @staticmethod
    def generate_index_vec(cad_vec):
        boundaries = torch.where(cad_vec[..., 0] == END_TOKEN.index("END_EXTRUSION"))[0]
        boundaries[-1] += 1
        return torch.bucketize(
            torch.arange(len(cad_vec), device=cad_vec.device), 
            boundaries=boundaries,
        ).to(cad_vec.dtype)
    
    @staticmethod
    def generate_flag_vec(cad_vec):
        dtype = cad_vec.dtype
        device = cad_vec.device

        flag_vec = torch.zeros(len(cad_vec), dtype=dtype, device=device)
        pattern = torch.arange(ONE_EXT_SEQ_LENGTH + 1, dtype=dtype, device=device)
        pattern[0] = 1

        starts = torch.where(cad_vec[..., 0] == END_TOKEN.index("END_SKETCH"))[0] + 1
        indices = starts[..., None] + torch.arange(ONE_EXT_SEQ_LENGTH + 1, device=device)

        flag_vec[indices] = pattern
        flag_vec[cad_vec[..., 0] == END_TOKEN.index("PADDING")] = ONE_EXT_SEQ_LENGTH + 1
        return flag_vec

    @staticmethod
    def generate_data(cad_vec):
        cad_vec = StartEnd.convert_vec(cad_vec.clone())
        cad_vec = torch.cat([
            cad_vec, 
            torch.zeros(
                (MAX_CAD_SEQUENCE_LENGTH - len(cad_vec), 2), 
                dtype=cad_vec.dtype,
                device=cad_vec.device,
            ),
        ], dim=0)
        vec_dict = {
            "cad_vec": cad_vec,
            "index_vec": StartEnd.generate_index_vec(cad_vec),
            "flag_vec": StartEnd.generate_flag_vec(cad_vec),
        }
        mask_cad_dict = {
            "attn_mask": StartEnd.generate_attention_mask(
                len(cad_vec) - 1, device=cad_vec.device
            ),
            "key_padding_mask": StartEnd.generate_key_padding_mask(cad_vec),
        }
        return vec_dict, mask_cad_dict


class Text2CAD_Dataset(Dataset):
    def __init__(
        self,
        cad_seq_dir: str,
        prompt_path: str,
        split_filepath: str,
        subset: str,
    ):
        """
        Args:
            cad_seq_dir (string): Directory with all the .pth files.
            prompt_path (string): Directory with all the .npz files.
            split_filepath (string): Train_Test_Val json file path.
            subset (string): "train", "test" or "val"
        """
        super(Text2CAD_Dataset, self).__init__()
        self.cad_seq_dir = cad_seq_dir
        self.prompt_path = prompt_path
        with open(prompt_path, "r") as f:
            self.prompt_data = json.load(f)

        self.all_prompt_choices = ["abstract", "beginner", "intermediate", "expert"]
        self.substrings_to_remove = ["*", "\n", '"', "\_", "\\", "\t", "-", ":"]

        # open spilt json
        with open(os.path.join(split_filepath), "r") as f:
            self.split = json.load(f)
        self.uid_pair = self.split[subset]
        self.subset = subset

        #list all uid keys
        self.samples = []
        self.build_samples()
        print(f"find {len(self.samples)} samples in {subset} data")

    def build_samples(self):
        for uid in self.uid_pair:
            if uid in self.prompt_data:
                for index, level_prompt in enumerate(self.prompt_data[uid]):
                    if index != self.all_prompt_choices.index("expert"):
                            continue
                    if isinstance(level_prompt, str):
                        self.samples.append((uid, index, self.get_vec_path(uid)))
                        self.prompt_data[uid][index] = self.remove_substrings(
                            text=level_prompt,
                            substrings=self.substrings_to_remove
                        ).lower()
                    else:
                        print(f"{uid} has no {self.all_prompt_choices[index]} prompt!")

    def get_vec_path(self, uid):
        root_id, chunk_id = uid.split("/")
        return os.path.join(self.cad_seq_dir, root_id, chunk_id, "seq", f"{chunk_id}.pth")

    def remove_substrings(self, text, substrings):
        """
        Remove specified substrings from the input text.

        Args:
            text (str): The input text to be cleaned.
            substrings (list): A list of substrings to be removed.

        Returns:
            str: The cleaned text with specified substrings removed.
        """
        # Escape special characters in substrings and join them to form the regex pattern
        regex_pattern = "|".join(re.escape(substring) for substring in substrings)
        # Use re.sub to replace occurrences of any substrings with an empty string
        cleaned_text = re.sub(regex_pattern, " ", text)
        # Remove extra white spaces
        cleaned_text = re.sub(" +", " ", cleaned_text)
        return cleaned_text

    def build_sketch_extrusion(self, vec_dict, mask_cad_dict):
        END_SKETCH_TOKEN_VAL = END_TOKEN.index("END_SKETCH")
        COORDINATE_TOKEN_VAL = END_PAD + BOOLEAN_PAD

        for key, value in vec_dict.items():
            vec_dict[key] = value[1:]
        
        indices_ext = torch.where(vec_dict["cad_vec"][:, 0] == END_SKETCH_TOKEN_VAL)[0] + 1
        n_ext = indices_ext.shape[0]

        mask_pt = vec_dict["cad_vec"][:, 1] >= COORDINATE_TOKEN_VAL
        mask_skt = torch.stack(
            [mask_pt & (vec_dict["index_vec"] == i_ext) for i_ext in range(n_ext)], dim=0)
        
        mask_cad_dict.update(
            {
                "n_extrusion": n_ext,
                "indices_extrusion": indices_ext,
                "mask_sketch_points": mask_skt,
            }
        )
        return mask_cad_dict

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        uid, level_index, cad_vec_path = self.samples[index]
        cad_vec_dict = torch.load(cad_vec_path, weights_only=True)
        prompt = self.prompt_data[uid][level_index]
        vec_dict, mask_cad_dict = cad_vec_dict["vec"], cad_vec_dict["mask_cad_dict"]

        if not ORIGINAL and self.subset not in ["test"]:
            vec_dict, mask_cad_dict = StartEnd.generate_data(vec_dict["cad_vec"])

        if self.subset in ["train"]:
            if not ORIGINAL:
                mask_cad_dict["shifted_key_padding_mask"] = (
                    (vec_dict["cad_vec"] == END_TOKEN.index("END_CURVE")) | \
                        mask_cad_dict["key_padding_mask"]
                )[1:]
            else:
                mask_cad_dict["shifted_key_padding_mask"] = \
                    mask_cad_dict["key_padding_mask"][1:]
            mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][:-1]

            self.build_sketch_extrusion(vec_dict.copy(), mask_cad_dict)

            cad_vec_target = vec_dict["cad_vec"][1:].clone()
            for key, value in vec_dict.items():
                vec_dict[key] = value[:-1]
            vec_dict["cad_vec_target"] = cad_vec_target

        return f"{uid}_{self.all_prompt_choices[level_index]}", \
            vec_dict, prompt, mask_cad_dict
    
    @staticmethod
    def collate_fn(batch):
        uid_level_list = [sample[0] for sample in batch]
        vec_dict_list = [sample[1] for sample in batch]
        prompt_list = [sample[2] for sample in batch]
        mask_cad_dict_list = [sample[3] for sample in batch]

        vec_dict_out = default_collate(vec_dict_list)

        mask_cad_dict_out = {}
        for key in mask_cad_dict_list[0].keys():
            if key in ["indices_extrusion", "mask_sketch_points"]:
                mask_cad_dict_out[key] = torch.cat(
                    [d[key] for d in mask_cad_dict_list], dim=0)
            elif key == "n_extrusion":
                mask_cad_dict_out[key] = torch.tensor([d[key] for d in mask_cad_dict_list])
            else:
                mask_cad_dict_out[key] = default_collate(
                    [d[key] for d in mask_cad_dict_list])
        
        return uid_level_list, vec_dict_out, prompt_list, mask_cad_dict_out


class Draw2CAD_Dataset(Dataset):
    def __init__(
        self,
        cad_seq_dir: str,
        svg_dir: str,
        split_filepath: str,
        subset: str,
        input_option: str,
    ):
        super(Draw2CAD_Dataset, self).__init__()
        self.cad_seq_dir = cad_seq_dir
        self.svg_dir = svg_dir

        self.input_option = input_option

        # open spilt json
        with open(os.path.join(split_filepath), "r") as f:
            self.split = json.load(f)
        self.uid_pair = self.split[subset]
        self.subset = subset

        #list all uid keys
        self.samples = [
            (uid, self.get_vec_path(uid), self.get_svg_path(uid))
            for uid in self.uid_pair
        ]
        print(f"find {len(self.samples)} samples in {subset} data")

    def get_vec_path(self, uid):
        root_id, chunk_id = uid.split("/")
        return os.path.join(
            self.cad_seq_dir, root_id, chunk_id, "seq", f"{chunk_id}.pth"
        )

    def get_svg_path(self, uid):
        root_id, chunk_id = uid.split("/")
        return os.path.join(
            self.svg_dir, root_id, f"{chunk_id}.npy",
        )

    def build_sketch_extrusion(self, vec_dict, mask_cad_dict):
        END_SKETCH_TOKEN_VAL = END_TOKEN.index("END_SKETCH")
        COORDINATE_TOKEN_VAL = END_PAD + BOOLEAN_PAD

        vec_dict = vec_dict.copy()
        for key, value in vec_dict.items():
            vec_dict[key] = value[1:]
        
        indices_ext = torch.where(vec_dict["cad_vec"][:, 0] == END_SKETCH_TOKEN_VAL)[0] + 1
        n_ext = indices_ext.shape[0]

        mask_pt = vec_dict["cad_vec"][:, 1] >= COORDINATE_TOKEN_VAL
        mask_skt = torch.stack(
            [mask_pt & (vec_dict["index_vec"] == i_ext) for i_ext in range(n_ext)], dim=0)
        
        mask_cad_dict.update(
            {
                "n_extrusion": n_ext,
                "indices_extrusion": indices_ext,
                "mask_sketch_points": mask_skt,
            }
        )
        return mask_cad_dict

    def __len__(self):
        return len(self.samples)

    def load_svg_data(self, file_path):
        data = np.load(file_path)

        # 1x
        if self.input_option == "1x":
            view_vec = data[300:, 0]
            command_vec = data[300:, 1]
            args_vec = data[300:, 2:]
        # 3x
        if self.input_option == "3x":
            view_vec = data[:300, 0]
            command_vec = data[:300, 1]
            args_vec = data[:300, 2:]
        # 4x
        if self.input_option == "4x":
            view_vec = data[:, 0]
            command_vec = data[:, 1]
            args_vec = data[:, 2:]

        view_vec = torch.tensor(view_vec, dtype=torch.long)
        command_vec = torch.tensor(command_vec, dtype=torch.long)
        args_vec = torch.tensor(args_vec, dtype=torch.long)
        
        return {"view": view_vec, "command": command_vec, "args": args_vec}

    def __getitem__(self, index):
        uid, cad_vec_path, svg_path = self.samples[index]

        cad_vec_dict = torch.load(cad_vec_path, weights_only=True)
        vec_dict, mask_cad_dict = cad_vec_dict["vec"], cad_vec_dict["mask_cad_dict"]

        svg_dict = self.load_svg_data(svg_path)

        if not ORIGINAL:
            vec_dict, mask_cad_dict = StartEnd.generate_data(vec_dict["cad_vec"])

        if self.subset in ["train"]:
            if not ORIGINAL:
                mask_cad_dict["shifted_key_padding_mask"] = (
                    (vec_dict["cad_vec"] == END_TOKEN.index("END_CURVE")) | \
                        mask_cad_dict["key_padding_mask"]
                )[1:]
            else:
                mask_cad_dict["shifted_key_padding_mask"] = \
                    mask_cad_dict["key_padding_mask"][1:]
            mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][:-1]

            self.build_sketch_extrusion(vec_dict, mask_cad_dict)

            cad_vec_target = vec_dict["cad_vec"][1:].clone()
            for key, value in vec_dict.items():
                vec_dict[key] = value[:-1]
            vec_dict["cad_vec_target"] = cad_vec_target

        return uid, vec_dict, mask_cad_dict, svg_dict

    @staticmethod
    def collate_fn(batch):
        uid_list = [sample[0] for sample in batch]
        vec_dict_list = [sample[1] for sample in batch]
        mask_cad_dict_list = [sample[2] for sample in batch]
        svg_dict_list = [sample[3] for sample in batch]

        uid_out = default_collate(uid_list)
        vec_dict_out = default_collate(vec_dict_list)
        svg_dict_out = default_collate(svg_dict_list)

        mask_cad_dict_out = {}
        for key in mask_cad_dict_list[0].keys():
            if key in ["indices_extrusion", "mask_sketch_points"]:
                mask_cad_dict_out[key] = torch.cat(
                    [d[key] for d in mask_cad_dict_list], dim=0
                )
            elif key == "n_extrusion":
                mask_cad_dict_out[key] = torch.tensor(
                    [d[key] for d in mask_cad_dict_list]
                )
            else:
                mask_cad_dict_out[key] = default_collate(
                    [d[key] for d in mask_cad_dict_list]
                )

        return uid_out, vec_dict_out, mask_cad_dict_out, svg_dict_out


def get_draw2cad_dataloaders(
    cad_seq_dir: str,
    svg_dir: str,
    split_filepath: str,
    subsets: list[str],
    input_option: str,
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    prefetch_factor: int,
):
    all_dataloaders = []

    for subset in subsets:
        is_train = subset == "train"

        # Create an instance of the Text2CADDataset
        dataset = Draw2CAD_Dataset(
            cad_seq_dir=cad_seq_dir,
            svg_dir=svg_dir,
            split_filepath=split_filepath,
            subset=subset,
            input_option=input_option,
        )

        if subset == "test":
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(
                dataset,
                shuffle=is_train,
            )

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory, # Set to True if using CUDA
            collate_fn=Draw2CAD_Dataset.collate_fn
        )
        all_dataloaders.append(dataloader)

    return all_dataloaders


def get_dataloaders(
    cad_seq_dir: str,
    prompt_path: str,
    split_filepath: str,
    subsets: list[str],
    batch_size: int,
    pin_memory: bool,
    num_workers: int,
    prefetch_factor: int,
):
    """
    Generate a DataLoader for the Text2CADDataset.

    Args:
    - cad_seq_dir (str): The directory containing the CAD sequence files.
    - prompt_path (str): The path to the CSV file containing the prompts.
    - split_filepath (str): The path to the JSON file containing the train/test/validation split.
    - subsets (list[str]): The subset to use ("train", "test", or "val").
    - batch_size (int): The batch size.
    - pin_memory (bool): Whether to pin memory.
    - num_workers (int): The number of workers.

    Returns:
    - dataloader (torch.utils.data.DataLoader): The DataLoader object.
    """

    all_dataloaders = []

    for subset in subsets:
        is_train = subset == "train"

        # Create an instance of the Text2CADDataset
        dataset = Text2CAD_Dataset(
            cad_seq_dir=cad_seq_dir,
            prompt_path=prompt_path,
            split_filepath=split_filepath,
            subset=subset,
        )

        if subset == "test":
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(
                dataset,
                shuffle=is_train,
            )

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory, # Set to True if using CUDA
            collate_fn=Text2CAD_Dataset.collate_fn
        )
        all_dataloaders.append(dataloader)

    return all_dataloaders