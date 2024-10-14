import csv
import json
import os
import random as rd
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from docetl.parsing_tools import get_parser, get_parsing_tools
from docetl.schemas import ParsingTool

from diskcache import Index
import mmap
from io import StringIO


DOCETL_HOME_DIR = os.path.expanduser("~/.docetl")

CACHE_DIR = os.path.join(DOCETL_HOME_DIR, "cache")
DATASET_CACHE_DIR = os.path.join(CACHE_DIR, "datasets")
DatasetsDiskCacheIndex = Index(dir=DATASET_CACHE_DIR)
DatasetsDiskCacheIndex.cache.close()


def create_parsing_tool_map(
    parsing_tools: Optional[List[ParsingTool]],
) -> Dict[str, ParsingTool]:
    """
    Create a mapping of parsing tool names to their corresponding ParsingTool objects.

    Args:
        parsing_tools (Optional[List[ParsingTool]]): A list of ParsingTool objects.

    Returns:
        Dict[str, ParsingTool]: A dictionary mapping tool names to ParsingTool objects.
    """
    if not parsing_tools:
        return {}

    if not isinstance(parsing_tools[0], ParsingTool):
        parsing_tools = [ParsingTool(**tool) for tool in parsing_tools]

    return {tool.name: tool for tool in parsing_tools}


class Dataset:
    """
    A class representing a dataset with various loading and parsing capabilities.

    Attributes:
        type (str): The type of the dataset ('file' or 'memory').
        source (str): The source of the dataset (currently only 'local' is supported).
        path_or_data (Union[str, List[Dict]]): The file path or in-memory data.
        parsing (List[Dict[str, str]]): A list of parsing tools to apply to the data.
        user_defined_parsing_tool_map (Dict[str, ParsingTool]): A map of user-defined parsing tools.
    """

    def __init__(
        self,
        runner,
        type: str,
        path_or_data: Union[str, List[Dict]],
        source: str = "local",
        parsing: List[Dict[str, str]] = None,
        user_defined_parsing_tool_map: Dict[str, ParsingTool] = {},
    ):
        """
        Initialize a Dataset object.

        Args:
            type (str): The type of the dataset ('file' or 'memory').
            source (str): The source of the dataset (currently only 'local' is supported).
            path_or_data (Union[str, List[Dict]]): The file path or in-memory data.
            parsing (List[Dict[str, str]], optional): A list of parsing tools to apply to the data.
            user_defined_parsing_tool_map (Dict[str, ParsingTool], optional): A map of user-defined parsing tools.
        """
        self.runner = runner
        self.type = self._validate_type(type)
        self.source = self._validate_source(source)
        self.path_or_data = self._validate_path_or_data(path_or_data)
        self.parsing = self._validate_parsing(parsing)
        self.user_defined_parsing_tool_map = user_defined_parsing_tool_map
        self.file_data = None
        self.data_hash = None
        self.file_metadata = {"size": None, "mtime": None}

    def _validate_type(self, type: str) -> str:
        """
        Validate the dataset type.

        Args:
            type (str): The type to validate.

        Returns:
            str: The validated type.

        Raises:
            ValueError: If the type is not 'file' or 'memory'.
        """
        if type not in ["file", "memory"]:
            raise ValueError("Type must be 'file' or 'memory'")
        return type

    def _validate_source(self, source: str) -> str:
        """
        Validate the dataset source.

        Args:
            source (str): The source to validate.

        Returns:
            str: The validated source.

        Raises:
            ValueError: If the source is not 'local'.
        """
        if source != "local":
            raise ValueError("Source must be 'local'")
        return source

    def _validate_path_or_data(
        self, path_or_data: Union[str, List[Dict]]
    ) -> Union[str, List[Dict]]:
        """
        Validate the path or data of the dataset.

        Args:
            path_or_data (Union[str, List[Dict]]): The path or data to validate.

        Returns:
            Union[str, List[Dict]]: The validated path or data.

        Raises:
            ValueError: If the path or data is invalid for the given type.
        """
        if self.type == "file":
            if not isinstance(path_or_data, str):
                raise ValueError("For type 'file', path_or_data must be a string")
            valid_extensions = (".json", ".csv")
            if not path_or_data.lower().endswith(valid_extensions):
                raise ValueError(f"Path must end with one of {valid_extensions}")
        elif self.type == "memory":
            if not isinstance(path_or_data, list):
                raise ValueError(
                    "For type 'memory', path_or_data must be a list of dictionaries"
                )
        return path_or_data

    def _validate_parsing(
        self, parsing_tools: Union[List[Dict[str, str]], None]
    ) -> List[Dict[str, str]]:
        """
        Validate the parsing tools.

        Args:
            parsing_tools (Union[List[Dict[str, str]], None]): The parsing tools to validate.

        Returns:
            List[Dict[str, str]]: The validated parsing tools.

        Raises:
            ValueError: If any parsing tool is invalid.
        """
        if parsing_tools is None:
            return []

        for tool in parsing_tools:
            if not isinstance(tool, dict) or "function" not in tool:
                raise ValueError(
                    "Each parsing tool must be a dictionary with a 'function' key and any arguments required by that function"
                )
            if not isinstance(tool["function"], str):
                raise ValueError("'function' in parsing tools must be a string")
            if "function_kwargs" in tool and not isinstance(
                tool["function_kwargs"], dict
            ):
                raise ValueError("'function_kwargs', if present, must be a dictionary")

        return parsing_tools

    def __repr__(self):
        """
        Return a string representation of the Dataset object.

        Returns:
            str: A string representation of the Dataset object.
        """
        return f"Dataset(type='{self.type}', source='{self.source}', path_or_data='{self.path_or_data}', parsing={self.parsing})"

    def _compute_hash(self, data: List[Dict]) -> int:
        hasher = hashlib.md5()
        for item in data:
            # Sort the keys to ensure consistent ordering
            sorted_item = json.dumps(item, sort_keys=True)
            hasher.update(sorted_item.encode("utf-8"))
        return int(hasher.hexdigest(), 16)

    def _dataloader(self) -> int:
        if self.type == "memory":
            self.data_hash = self._compute_hash(self.path_or_data)
            return self.data_hash

        file_stat = os.stat(self.path_or_data)
        current_size = file_stat.st_size
        current_mtime = file_stat.st_mtime

        if (
            self.file_metadata["size"] == current_size
            and self.file_metadata["mtime"] == current_mtime
            and self.file_data is not None
        ):
            return self._compute_hash(self.file_data)

        _, ext = os.path.splitext(self.path_or_data.lower())

        if ext == ".json":
            self._load_json(self.path_or_data)
        elif ext == ".csv":
            self._load_csv(self.path_or_data)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self.file_metadata["size"] = current_size
        self.file_metadata["mtime"] = current_mtime
        self.data_hash = self._compute_hash(self.file_data)
        return self.data_hash

    def _apply_parsing_tools(self) -> List[Dict]:
        if self.data_hash is None:
            self._dataloader()

        assert self.data_hash is not None, "Data hash is not set"

        cached_data = DatasetsDiskCacheIndex.get(self.data_hash)
        if cached_data is not None:
            return cached_data

        data = self.path_or_data if self.type == "memory" else self.file_data

        if not data or not self.parsing:
            return data

        def process_item(item, tools):
            output = []
            for tool in tools:
                function_kwargs = dict(tool)
                function_kwargs.pop("function")
                if "function_kwargs" in function_kwargs:
                    function_kwargs.update(function_kwargs.pop("function_kwargs"))

                try:
                    func = get_parser(tool["function"])
                except KeyError:
                    if (
                        self.user_defined_parsing_tool_map
                        and tool["function"] in self.user_defined_parsing_tool_map
                    ):
                        # Define the custom function in the current scope
                        exec(
                            "from typing import List, Dict\n"
                            + self.user_defined_parsing_tool_map[
                                tool["function"]
                            ].function_code
                        )
                        # Get the function object
                        func = locals()[tool["function"]]
                    else:
                        raise ValueError(
                            f"Parsing tool {tool['function']} not found. Please define it or use one of our existing parsing tools: {get_parsing_tools()}"
                        )

                result = func(item, **function_kwargs)
                output.extend([item | res for res in result])
            return output

        parsed_data = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_item, item, self.parsing) for item in data
            ]
            for future in as_completed(futures):
                parsed_data.extend(future.result())

        assert len(parsed_data) > 0, "No data was returned from parsing tools"
        DatasetsDiskCacheIndex[self.data_hash] = parsed_data
        return parsed_data

    def load(self) -> List[Dict]:
        """
        Load the dataset from the specified path or return the in-memory data.

        Returns:
            List[Dict]: A list of dictionaries representing the dataset.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        self._dataloader()
        return self._apply_parsing_tools()

    def sample(self, n: int, random_sample: bool = True) -> List[Dict]:
        """
        Sample n items from the parsed dataset.

        Args:
            n (int): Number of items to sample.
            random_sample (bool): If True, sample randomly. If False, take the first n items.

        Returns:
            List[Dict]: A list of n sampled items.

        Raises:
            ValueError: If the sample size is larger than the dataset size.
        """
        parsed_data = self.load()  # This will use cached data if available

        if n > len(parsed_data):
            raise ValueError(
                f"Sample size {n} is larger than dataset size {len(parsed_data)}"
            )

        if random_sample:
            return rd.sample(parsed_data, n)
        else:
            return parsed_data[:n]

    def _load_csv(self, path: str):
        with open(path, "r+b") as f:
            # Memory-map the file for faster access
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Use StringIO to create a file-like object from the memory-mapped file
            csv_data = StringIO(mm.read().decode("utf-8"))

            # Use csv.DictReader for efficient parsing
            reader = csv.DictReader(csv_data)

            # Use a generator expression instead of a list comprehension
            self.file_data = (dict(row) for row in reader)

            # Close the memory-map
            mm.close()

    def _load_json(self, path: str):
        with open(path, "r+b") as f:
            # Memory-map the file for faster access
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Use StringIO to create a file-like object from the memory-mapped file
            json_data = StringIO(mm.read().decode("utf-8"))

            # Use json.load for parsing
            self.file_data = json.load(json_data)

            # Close the memory-map
            mm.close()
