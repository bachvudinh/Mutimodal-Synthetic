from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class Caption:
    caption: str

@dataclass
class Box:
    category_name: str
    bbox: List[float]  # normalized [min_x, min_y, max_x, max_y]
    confidence: Optional[float]

@dataclass
class Context:
    sample_id: str
    source: str
    captions: List[Caption]
    boxes: Optional[List[Box]] = None

@dataclass
class SampleResponse:
    instruction: str
    response: str

@dataclass
class Sample:
    id: str
    instruction: str
    response: str
    image: str
    image_source: str
    type: str

    {"id": "000000525439", "image": "COCO_val2014_000000525439.jpg",
     "instruction": "What is the position of the skateboard in the image?",
     "output": "The skateboard in the image is in an upside-down position, with its wheels pointing up and laying on the ground.",
     "type": "conv"}
