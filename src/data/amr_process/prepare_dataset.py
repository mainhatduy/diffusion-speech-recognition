import json
import string
import sys
import logging
import os

# Cấu hình logging để dễ theo dõi trên Fedora terminal
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Tắt logger của Penman để giảm output
logging.getLogger("penman").setLevel(logging.WARNING)

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

from collections import Counter
from enum import auto
from os import PathLike
from pathlib import Path
from typing import List, Union, Optional

import pandas as pd
import penman
from datasets import Dataset, DatasetDict, load_dataset
from ftfy import fix_text
from .postprocessing_str import postprocess_str_after_linearization

try:
    from multi_amr.utils import get_penman_model, remove_wiki_from_graph
except ImportError:
    # Fallback nếu không có thư viện custom, dùng model chuẩn
    logger.warning("Không tìm thấy multi_amr.utils, sử dụng Penman model mặc định.")
    from penman.model import Model
    
    def get_penman_model(dereify=False):
        # Return Model instance, không phải module
        return Model()
    
    def remove_wiki_from_graph(graph):
        # Loại bỏ các triplet có :wiki từ AMR graph
        new_triples = []
        for triple in graph.triples:
            # triple có dạng (source, role, target)
            # Bỏ qua các triplet có role là :wiki
            if triple[1] != ':wiki':
                new_triples.append(triple)
        
        # Tạo graph mới từ các triplet đã lọc
        return penman.Graph(new_triples)

from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tqdm import tqdm


class SplitType(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()

    @classmethod
    def from_string(cls, split_str):
        # Map tên split của HuggingFace sang Enum của code
        split_str = split_str.lower()
        if split_str in ("train", "training"):
            return cls.TRAIN
        elif split_str in ("validation", "dev", "val"):
            return cls.VALIDATION
        elif split_str == "test":
            return cls.TEST
        raise ValueError(
            f"'{split_str}' is not a valid {cls.__name__} value."
        )


def prepare_dataset(
    dataset_name: str,
    src_column: str,
    tgt_column: str,
    output_dir: Optional[Union[str, PathLike]] = None,
    dedupe: bool = False,
    remove_wiki: bool = False,
    fix_ftfy: bool = False,
    normalize_punct: bool = False,
    detokenize: bool = False,
    remove_bracketed: bool = False,
    dereify: bool = False,
    lang: str = "vi",
    cache_dir: Optional[str] = None,
):
    """
    Load dữ liệu từ HuggingFace, xử lý tiếng Việt (vi) và AMR,
    sau đó lưu lại đúng định dạng JSONL/Arrow mà code gốc yêu cầu.
    """
    if output_dir:
        pdout = Path(output_dir).resolve()
        pdout.mkdir(exist_ok=True, parents=True)
    
    # Khởi tạo các công cụ xử lý
    penman_model = get_penman_model(dereify=dereify)
    punct_norm_text = MosesPunctNormalizer(lang=lang)
    detokenizer = MosesDetokenizer(lang=lang)
    detokenize_func = detokenizer.detokenize

    logger.info(f"Đang tải dataset: {dataset_name}")
    # Load dataset từ HF
    try:
        raw_datasets = load_dataset(dataset_name, cache_dir=cache_dir)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        # Try loading from disk if it's a local path
        if os.path.exists(dataset_name):
             raw_datasets = load_from_disk(dataset_name)
        else:
            raise e
    
    # Cấu trúc lưu trữ data trước khi đưa vào DataFrame
    data = {
        "split_type": [],
        src_column: [], 
        tgt_column: []
    }
    
    print("Processing main...")
    
    # Duyệt qua các split trên HF (train, validation, test)
    for hf_split in raw_datasets.keys():
        try:
            split_type = SplitType.from_string(hf_split)
        except ValueError:
            logger.warning(f"Bỏ qua split lạ: {hf_split}")
            continue
            
        logger.info(f"Đang xử lý split: {split_type}")
        
        current_split_data = raw_datasets[hf_split]
        
        # Dùng tqdm để hiện thanh tiến trình
        for idx, sample in enumerate(tqdm(current_split_data, desc=f"Processing {split_type}")):
            
            # 1. Lấy câu tiếng Việt (vi)
            sentence = sample.get(src_column, "")
            if not sentence: 
                continue # Bỏ qua nếu câu rỗng

            # 2. Xử lý text (Pre-processing)
            if fix_ftfy:
                sentence = fix_text(sentence)
            if normalize_punct:
                sentence = punct_norm_text.normalize(sentence)
            if detokenize:
                # Lưu ý: Tiếng Việt tokenized thường dùng underscore (dùng_cơm), 
                # detokenize của Moses có thể không hoàn hảo cho TV, nhưng giữ logic cũ.
                sentence = detokenize_func(sentence.split())

            # 3. Lấy và xử lý AMR
            raw_amr = sample.get(tgt_column, "")
            if not raw_amr:
                continue

            try:
                # Parse AMR string thành Graph object
                graph = penman.decode(raw_amr, model=penman_model)
                
                if remove_wiki:
                    graph = remove_wiki_from_graph(graph)
                
                # For SPRING format, encode only the graph structure without metadata comments
                # This produces linearized AMR like: (a2 / arrive-01 :arg1 (p2 / person ...) ...)
                # instead of including # ::id and # ::snt comments
                clean_penman = penman.encode(
                    graph, 
                    model=penman_model, 
                    indent=None,  # Single line output
                    compact=False  # Keep spaces for readability
                ).replace("–", "-")
                
                # Remove any metadata comments that might have been included
                # Keep only the graph structure (the part starting with opening parenthesis)
                lines = clean_penman.split('\n')
                graph_lines = [line for line in lines if not line.strip().startswith('#')]
                clean_penman = ' '.join(graph_lines).strip()
                
                linearized = postprocess_str_after_linearization(clean_penman)
                
                # 4. Append vào danh sách
                data["split_type"].append(str(split_type))
                data[src_column].append(sentence)
                data[tgt_column].append(linearized)

            except penman.DecodeError:
                logger.warning(f"Lỗi parse AMR tại index {idx} của {split_type}. Bỏ qua.")
                continue
            except Exception as e:
                logger.error(f"Lỗi không xác định tại index {idx}: {e}")
                continue

    # Chuyển sang DataFrame
    df = pd.DataFrame(data)
    del data

    print("\nExample data (before filtering):")
    print(df.head(3))

    processing_info = {
        "dataset_name": dataset_name,
        "lang": lang,
        "dedupe": dedupe,
        "remove_wiki": remove_wiki,
        "fix_ftfy": fix_ftfy,
        "normalize_punct": normalize_punct,
        "detokenize": detokenize,
        "remove_bracketed": remove_bracketed,
    }

    # Lọc các câu bắt đầu/kết thúc bằng dấu câu (như code cũ)
    if remove_bracketed:
        def starts_ends_with_punctuation(s):
            return s.startswith(tuple(string.punctuation)) and s.endswith(tuple(string.punctuation))
        
        df = df[~df[src_column].apply(starts_ends_with_punctuation)]

    # Lọc trùng lặp
    if dedupe:
        df_len_before = len(df.index)
        df.drop_duplicates(subset=[src_column], inplace=True)
        print(f"Dropped {(df_len_before - len(df.index)):,} duplicates!")
        processing_info["num_dropped_duplicates"] = df_len_before - len(df.index)

    # Tạo DatasetDict và lưu file
    datasets_dict = DatasetDict()
    processing_info["final_sizes"] = {}
    
    for split_type, groupdf in df.groupby("split_type"):
        # Xóa cột split_type vì nó đã nằm trong key của DatasetDict
        groupdf_clean = groupdf.drop(columns=["split_type"])
        
        # Chuyển về HF Dataset object
        hf_ds = Dataset.from_pandas(groupdf_clean)
        
        # Lưu ý: drop index cột do pandas tạo ra (nếu có)
        if "__index_level_0__" in hf_ds.column_names:
            hf_ds = hf_ds.remove_columns(["__index_level_0__"])

        datasets_dict[split_type] = hf_ds
        
        if output_dir:
            # Lưu JSONL (để tương thích code cũ)
            jsonl_path = pdout.joinpath(f"{split_type}.jsonl")
            hf_ds.to_json(jsonl_path, force_ascii=False) # force_ascii=False để giữ tiếng Việt
            
            print(f"Processed {split_type} set containing {len(groupdf):,} samples!")
            processing_info["final_sizes"][split_type] = len(groupdf)

    if output_dir:
        # Lưu DatasetDict (Arrow format)
        datasets_dict.save_to_disk(str(pdout))
        
        # Lưu info log
        pdout.joinpath("processing_info.json").write_text(json.dumps(processing_info, indent=4), encoding="utf-8")
        logger.info(f"Hoàn tất! Dữ liệu đã lưu tại: {pdout}")
    
    return datasets_dict



def main():
    import argparse

    cparser = argparse.ArgumentParser(description="Prepare HF AMR Dataset for Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cparser.add_argument(
        "--dataset_name",
        default="myduy/amr-translated",
        help="Hugging Face dataset name containing 'vi' and 'amr' columns",
    )
    cparser.add_argument(
        "--src_column",
        default="vi",
        help="Column name for source language (e.g., 'vi')",
    )
    cparser.add_argument(
        "--tgt_column",
        default="amr",
        help="Column name for target language (e.g., 'amr')",
    )
    cparser.add_argument("-o", "--output_dir", required=True, help="dir to write the dataset files to")
    cparser.add_argument(
        "--dedupe",
        action="store_true",
        help="whether to deduplicate the data",
    )
    cparser.add_argument(
        "--remove_wiki",
        action="store_true",
        help="whether to remove wiki from the AMR entries",
    )
    cparser.add_argument(
        "--fix_ftfy",
        action="store_true",
        help="whether to fix text issues",
    )
    cparser.add_argument(
        "--normalize_punct",
        action="store_true",
        help="whether to normalize punctuation",
    )
    cparser.add_argument(
        "--detokenize",
        action="store_true",
        help="whether to detokenize",
    )
    cparser.add_argument(
        "--remove_bracketed",
        action="store_true",
        help="whether to remove sentences that start and end with punctuation",
    )
    
    cargs = cparser.parse_args()
    prepare_dataset(**vars(cargs))


if __name__ == "__main__":
    main()