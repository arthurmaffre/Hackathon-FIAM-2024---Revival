import os
import yaml
from typing import List, Dict, Callable, Any
import importlib
import ast

################################
#  Author: Thomas Vaudescal    #
#  Date: December 18, 2023     #
################################


class ConfigReader:
    def __init__(self, config_paths: List[str]) -> None:
        if not config_paths:
            raise ValueError("At least one configuration file path must be provided.")
        self.config_paths = config_paths
        self.configs = self._load_configs()

    @staticmethod
    def _yaml_function_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> Callable:
        """Convert a YAML node with !function tag into a Python function."""
        func_name = loader.construct_scalar(node=node)
        function = ConfigReader._import_function(module_name='custom_functions', function_name=func_name)
        if function is None:
            raise ValueError(f"Function '{func_name}' is not available.")
        return function

    @staticmethod
    def _import_function(module_name: str, function_name: str) -> Callable:
        """Dynamically imports a function from a given module."""
        try:
            module = importlib.import_module(name=module_name)
            return getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Function '{function_name}' could not be imported from module '{module_name}'.") from e

    @staticmethod
    def _convert_str_to_tuple(data: Any) -> Any:
        """Recursively converts strings that represent tuples back into tuples."""
        if isinstance(data, str) and data.startswith('(') and data.endswith(')'):
            try:
                # Safely evaluate the string as a tuple
                return ast.literal_eval(data)
            except (SyntaxError, ValueError):
                return data  # Return the original string if it's not a valid tuple
        elif isinstance(data, dict):
            return {k: ConfigReader._convert_str_to_tuple(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [ConfigReader._convert_str_to_tuple(i) for i in data]
        else:
            return data

    @staticmethod
    def _load_yaml(file_path: str) -> Dict[str, Any]:
        """Loads a YAML file, converts strings to tuples where necessary, and returns its content."""
        try:
            with open(file_path, 'r') as file:
                loader = yaml.SafeLoader
                loader.add_constructor('!function', ConfigReader._yaml_function_constructor)
                config = yaml.load(stream=file, Loader=loader)
                return ConfigReader._convert_str_to_tuple(config)
        except FileNotFoundError:
            raise FileNotFoundError(f"The specified file '{file_path}' was not found.")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error reading YAML file: {e}")

    def _load_configs(self) -> List[Dict[str, Any]]:
        """Loads the configurations from the specified paths."""
        configs = []
        for config_path in self.config_paths:
            if not os.path.exists(path=config_path):
                raise FileNotFoundError(f"The configuration file at '{config_path}' was not found.")
            configs.append(self._load_yaml(file_path=config_path))
        return configs

    @staticmethod
    def _merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges two dictionaries."""
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                dict1[key] = ConfigReader._merge_dicts(dict1[key], value)
            else:
                dict1[key] = value
        return dict1

    @staticmethod
    def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merges multiple configuration dictionaries recursively."""
        merged_config = {}
        for config in configs:
            merged_config = ConfigReader._merge_dicts(merged_config, config)
        return merged_config

    def get_merged_config(self) -> Dict[str, Any]:
        """Returns the merged configuration."""
        return self.merge_configs(self.configs)


def print_separator(message: str) -> None:
    """
    Prints a stylized separator with the given message centered.

    Args:
        message (str): The message to be displayed within the separator.
    """
    separator_line = "═" * 2 + "≡" * 36 + "✦✦✦" + "≡" * 36 + "═" * 2
    total_width = len(separator_line)

    # Calculate the amount of padding needed on each side of the message
    message_length = len(message)
    padding_each_side = (total_width - 2 - message_length) // 2
    extra_padding = (total_width - 2 - message_length) % 2  # In case the message length is odd

    # Create the title lines with borders
    title_line_1 = "╔" + " " * (total_width - 2) + "╗"
    title_line_2 = (
            "║"
            + " " * padding_each_side
            + message
            + " " * (padding_each_side + extra_padding)
            + "║"
    )
    title_line_3 = "╚" + " " * (total_width - 2) + "╝"

    # Print the separator with title lines
    print(f"\n{separator_line}\n{title_line_1}\n{title_line_2}\n{title_line_3}\n{separator_line}\n")


def get_merged_config(config_paths: List[str], strategy_name: str) -> Dict[str, Any]:
    read_config = ConfigReader(config_paths=config_paths)
    print_separator(f"Strategy selected: {strategy_name}")
    return read_config.get_merged_config()


if __name__ == '__main__':
    config_paths = ['../../config/meta_config/classification_pipeline.yaml',
                    '../../config/strategy_config/ada_boost_classifier.yaml',
                    '../../config/meta_config/prediction_pipeline.yaml']

    config = get_merged_config(config_paths=config_paths, strategy_name="prediction_pipeline")
    print(config)
    print(config["prediction_pipeline"]["bayes_search_params_grid"])
