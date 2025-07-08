from llama_index.core.base.embeddings.base import BaseEmbedding

"""Jina embeddings file."""

from typing import Any, List, Optional, Dict
from urllib.parse import urlparse
from os.path import exists
import base64
import requests
import numpy as np

from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType

# The new V4 API supports a much larger batch size
MAX_BATCH_SIZE = 8192

DEFAULT_JINA_AI_API_URL = "https://api.jina.ai/v1"

# The new API uses "float" for all standard embeddings. Binary encodings are less common now.
VALID_ENCODING = ["float"]


class _JinaAPICaller:
    """Internal helper class to handle Jina API calls with the new format."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_JINA_AI_API_URL,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.api_url = f"{base_url}/embeddings"
        self.api_key = get_from_param_or_env("api_key", api_key, "JINAAI_API_KEY", "")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    def get_embeddings(
        self,
        input_list: List[Dict[str, str]], # Expects the new format: [{"text": "..."}, {"image": "..."}]
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """Get embeddings using the new Jina v4 API format."""
        # Call Jina AI Embedding API
        input_json = {
            "input": input_list,
            "model": self.model,
        }
        if task is not None:
            input_json["task"] = task
        if dimensions is not None:
            input_json["dimensions"] = dimensions

        resp = self._session.post(self.api_url, json=input_json).json()

        if "data" not in resp:
            raise RuntimeError(resp.get("detail", "Unknown error from Jina API"))

        embeddings = resp["data"]
        # Sort resulting embeddings by index to maintain original order
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])
        return [result["embedding"] for result in sorted_embeddings]

    async def aget_embeddings(
        self,
        input_list: List[Dict[str, str]],
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """Asynchronously get embeddings."""
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        input_json = {
            "input": input_list,
            "model": self.model,
        }
        if task is not None:
            input_json["task"] = task
        if dimensions is not None:
            input_json["dimensions"] = dimensions

        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                self.api_url, json=input_json, headers=headers
            ) as response:
                resp = await response.json()
                if not response.ok:
                    raise RuntimeError(resp.get("detail", "Unknown error from Jina API"))
                
                embeddings = resp["data"]
                sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])
                return [result["embedding"] for result in sorted_embeddings]


def _is_local(url: str) -> bool:
    """Check if a URL points to a local file."""
    url_parsed = urlparse(url)
    return url_parsed.scheme in ("file", "") and exists(url_parsed.path)


def _get_bytes_str(file_path: str) -> str:
    """Read a local file and return its base64 encoded string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class JinaEmbedding(MultiModalEmbedding):
    """
    JinaAI class for embeddings, updated for the v4 API.

    This class supports text, image, and mixed-modal embedding using the new
    `jina-embeddings-v4` format.

    Args:
        model (str): Model for embedding. Defaults to `jina-embeddings-v4`.
        api_key (str): The JinaAI API key.
        task (str): The task type for the embedding, e.g., "text-matching", "retrieval".
        dimensions (int): The desired dimension for the output embeddings.
    """

    api_key: Optional[str] = Field(default=None, description="The JinaAI API key.")
    model: str = Field(
        default="jina-embeddings-v4",
        description="The model to use when calling the Jina AI API.",
    )
    task: Optional[str] = Field(
        default=None, description="The task type for the embedding."
    )
    dimensions: Optional[int] = Field(
        default=None, description="The desired dimension of the output embeddings."
    )

    _api: _JinaAPICaller = PrivateAttr()

    def __init__(
        self,
        model: str = "jina-embeddings-v4",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if embed_batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {embed_batch_size} exceeds Jina's max batch size of {MAX_BATCH_SIZE}.")
            
        # Initialize the parent class
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model=model,
            api_key=api_key,
            task=task,
            dimensions=dimensions,
            **kwargs,
        )

        # Initialize the API caller with the provided credentials and model
        self._api = _JinaAPICaller(model=self.model, api_key=self.api_key)

    @classmethod
    def class_name(cls) -> str:
        return "JinaEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return await self._aget_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        result = await self._aget_text_embeddings([text])
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings for a list of texts."""
        input_list = [{"text": text} for text in texts]
        return self._api.get_embeddings(
            input_list=input_list,
            task=self.task,
            dimensions=self.dimensions,
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings for a list of texts."""
        input_list = [{"text": text} for text in texts]
        return await self._api.aget_embeddings(
            input_list=input_list,
            task=self.task,
            dimensions=self.dimensions,
        )

    def _get_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """Get image embedding."""
        return self._get_image_embeddings([img_file_path])[0]

    async def _aget_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """Asynchronously get image embedding."""
        embeddings = await self._aget_image_embeddings([img_file_path])
        return embeddings[0]

    def _get_image_embeddings(self, img_file_paths: List[ImageType]) -> List[List[float]]:
        """Get image embeddings for a list of image paths."""
        input_list = []
        for img_path in img_file_paths:
            if _is_local(img_path):
                input_list.append({"image": _get_bytes_str(img_path)})
            else:
                input_list.append({"image": img_path})
        return self._api.get_embeddings(input_list, task=self.task, dimensions=self.dimensions)

    async def _aget_image_embeddings(self, img_file_paths: List[ImageType]) -> List[List[float]]:
        """Asynchronously get image embeddings for a list of image paths."""
        input_list = []
        for img_path in img_file_paths:
            if _is_local(img_path):
                input_list.append({"image": _get_bytes_str(img_path)})
            else:
                input_list.append({"image": img_path})
        return await self._api.aget_embeddings(input_list, task=self.task, dimensions=self.dimensions)
  
