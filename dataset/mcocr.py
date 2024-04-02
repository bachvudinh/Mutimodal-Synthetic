import dataclasses
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Iterator

from tqdm import tqdm

from data_definitions import *


class MCOCRLoader:
    def __init__(self, storage_path: str = '.'):
        self.data_dir: Path = Path(storage_path) / "mcocr_train_df_rotate_filtered.json"
        self.captions: Dict[str, List[Caption]] = self._generate_captions()
        self.length = len(self.captions)

    def _generate_captions(self) -> Dict[str, List[Caption]]:

        captions: Dict[str, List[Caption]] = defaultdict(list)
        with open(self.data_dir) as f:
            data = json.load(f)
            for item in data:
                image_id = item['image_id']
                caption = item['texts']
                captions[image_id].append(Caption(caption=caption))

        return captions

    @property
    def name(self) -> str:
        return "mc-ocr"

    def __iter__(self) -> Iterator[Context]:
        for image_id, captions in self.captions.items():
            yield Context(sample_id=image_id, source=self.name, captions=captions)

    def __len__(self) -> int:
        return self.length
