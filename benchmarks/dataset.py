# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A simple ShareGPT dataset implementation for benchmarking.
# Reference: https://github.com/vllm-project/vllm/blob/6bf3b46d7840364907405c7b02eaa66886a90839/vllm/benchmarks/datasets.py#L1210

import base64
import io
import json
import logging
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image


logger = logging.getLogger(__name__)


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: str | list[str]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: dict | list[dict] | None = None
    request_id: str | None = None


class ShareGPTDataset:
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    Data source: Aeala/ShareGPT_Vicuna_unfiltered
    """

    DEFAULT_SEED = 0
    IS_MULTIMODAL = True

    def __init__(
        self,
        dataset_path: str | None = None,
        random_seed: int = DEFAULT_SEED,
        disable_shuffle: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.

        Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
                indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
                sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.disable_shuffle = disable_shuffle
        self.data = None

        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [entry for entry in self.data if "conversations" in entry and len(entry["conversations"]) >= 2]
        random.seed(self.random_seed)
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(self.data)

    def sample(
        self,
        tokenizer,
        num_requests: int,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        ind = 0
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt, completion = (
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
            )

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = len(completion_ids) if output_len is None else output_len
            if not is_valid_sequence(
                prompt_len,
                new_output_len,
                skip_min_output_len_check=output_len is not None,
            ):
                continue
            if image_path := entry.get("image"):
                mm_content = process_image(image_path)
            elif video_path := entry.get("video"):
                mm_content = process_video(video_path)
            else:
                mm_content = None
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(prompt, mm_content)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(ind),
                )
            )
            ind += 1
        self.maybe_oversample_requests(samples, num_requests, request_id_prefix, no_oversample)
        return samples

    def maybe_oversample_requests(
        self,
        requests: list[SampleRequest],
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
    ) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
                requests.
            num_requests (int): The target number of requests.
            request_id_prefix (str): The prefix applied to generated request
                identifiers.

        """
        if no_oversample:
            logger.info("Skipping oversampling. Total samples: %d.", len(requests))
            return

        if len(requests) < num_requests:
            random.seed(self.random_seed)
            needed = num_requests - len(requests)
            additional = []
            for i in range(needed):
                req = deepcopy(random.choice(requests))
                req.request_id = request_id_prefix + str(len(requests) + i)
                additional.append(req)
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.", num_requests)

        ids = [req.request_id for req in requests]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate request_id found in the sampled requests. Please ensure that each request_id is unique."
            )


def process_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG in memory.  - Encodes the JPEG data as a base64 string.  - Returns
       a dictionary with the image as a base64 data URL.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, Image.Image):
        image = convert_image_mode(image, "RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }

    if isinstance(image, str):
        image_url = image if image.startswith(("http://", "https://", "file://")) else f"file://{image}"
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image}. Must be a PIL.Image.Image or str or dictionary with raw image bytes."
    )


def process_video(video: Any) -> Mapping[str, Any]:
    """
    Process a single video input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw video bytes: - Expects a dict with a 'bytes' key
       containing raw video data.

    2. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(video, dict) and "bytes" in video:
        video_bytes = video["bytes"]
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        return {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
        }

    if isinstance(video, str):
        video_url = video if video.startswith(("http://", "https://", "file://")) else f"file://{video}"
        return {"type": "video_url", "video_url": {"url": video_url}}

    raise ValueError(
        f"Invalid video input {video}. Must be a string of local path/remote url, or a dictionary with raw video bytes in the form of `{{'bytes': raw_video_bytes}}`."  # noqa: E501
    )


def rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def convert_image_mode(image: Image.Image, to_mode: str):
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return rgba_to_rgb(image)
    else:
        return image.convert(to_mode)


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long or combined_too_long)
