import json
import os
import subprocess
from typing import Any, Callable, Dict, List

import shutil

from src.utils.NoIndentEncoder import NoIndentEncoder, noIndent_dictOrList_onFirstLevel


def read_json(filepath: str) -> Dict:
    """
    Args:
        filepath:

    Returns:
        json
    """
    with open(filepath, "r") as f:
        return json.loads(f.read())


def read_jsonl(filepath: str) -> List[Dict]:
    """
    Args:
        filepath:

    Returns:
        json_lines
    """
    json_lines = []

    with open(filepath, "r") as f:
        for idx, line in enumerate(f.readlines()):
            json_lines.append(json.loads(line.strip("\n")))

    return json_lines


def write_jsonl(json_toWrite: List[Dict], filepath: str):
    """
    Args:
        json_toWrite:
        filepath:
    """
    with open(filepath, "w+") as f:
        for json_toWrite in json_toWrite:
            f.write(json.dumps(json_toWrite))
            f.write("\n")


def write_json(json_toWrite: Dict, filepath: str):
    """
    Args:
        json_toWrite:
        filepath:
    """
    with open(filepath, "w+") as f:
        f.write(json.dumps(json_toWrite))


def append_jsonl(json_toWrite: List[Dict], filepath: str):
    """
    Args:
        json_toWrite:
        filepath:
    """
    with open(filepath, "a+") as f:
        f.write(json.dumps(json_toWrite))
        f.write("\n")


def append_json(json_toWrite: Dict, filepath: str, pretty_print: bool):
    """
    Args:
        json_toWrite:
        filepath:
        pretty_print:
    """
    with open(filepath, "a+") as f:
        if pretty_print:
            dumped_json = json.dumps(
                noIndent_dictOrList_onFirstLevel(json_toWrite),
                cls=NoIndentEncoder,
                indent=2,
            )
        else:
            dumped_json = json.dumps(json_toWrite)
        f.write(dumped_json + "\n")


def saveTo_gcp(
    should_saveToGCP: bool, source_filepath: str, destination_filepath: str = None
):
    """

    Args:
        should_saveToGCP:
        filepath:

    Returns:

    """
    if destination_filepath is None:
        destination_filepath = source_filepath

    if should_saveToGCP:
        subprocess.call(
            f"gsutil "
            f"-m "
            f"-o GSUtil:parallel_composite_upload_threshold=150M "
            f"cp -r {source_filepath} gs://realistic_evaluation_of_merging/{destination_filepath}",
            shell=True,
        )


def syncWith_gcp(should_saveToGCP: bool, filepath: str):
    """

    Args:
        should_saveToGCP:
        filepath:

    Returns:

    """
    if should_saveToGCP:
        subprocess.call(
            f"gsutil "
            f"-m "
            f"-o GSUtil:parallel_composite_upload_threshold=150M "
            f"rsync -r {filepath} gs://realistic_evaluation_of_merging/{filepath}",
            shell=True,
        )


def deleteFiles_inDirectory(directory: str, start_string):
    """

    Args:
        directory:
        start_string
    """
    for file in os.listdir(directory):
        if file.startswith(start_string):
            filepath = os.path.join(directory, file)
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                shutil.rmtree(filepath)


def getFile_inDirectory(directory: str, start_string):
    """

    Args:
        directory:
        start_string
    """
    for file in os.listdir(directory):
        if file.startswith(start_string):
            return file


def get_subdirectories(directory):
    subdirectories = [f.path for f in os.scandir(directory) if f.is_dir()]
    return subdirectories
