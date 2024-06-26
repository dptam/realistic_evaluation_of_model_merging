from typing import Any, Callable, Dict, List

import torch


def reduce_gatheredOutput(
    gathered_output: Dict, reduce_fn: Callable = None
) -> Dict[Any, Any]:
    """
    Reduces the output from multiple devices to have the same format as for a single device.

    Args:
        gathered_output:
        reduce_fn:

    Returns:
    """
    reduced_output = {}

    # Combine the values across the different devices
    for iterate_dict in gathered_output:
        for k, v in iterate_dict.items():
            if k in reduced_output:
                reduced_output[k].append(v)
            else:
                reduced_output[k] = [v]

    # Reduce the gathered output at each key
    for k, batch_ofValues in reduced_output.items():
        if isinstance(batch_ofValues[0], list):
            reduced_output[k] = [item for sublist in batch_ofValues for item in sublist]
        else:
            reduced_output[k] = [item for item in batch_ofValues]

        if reduce_fn is not None:
            reduced_output[k] = reduce_fn(reduced_output[k])

    return reduced_output


def is_nodeZero(device):
    """
    Args:
        device:

    Returns:
        whether it is zero node
    """
    return device == 0 or device == torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def is_distributedSetup(world_size: int):
    """
    Args:
        world_size:

    Returns:
        whether setup is distributed
    """
    return world_size != 1


def addDistributedArgs_toParser(parser):
    """

    Args:
        parser:

    Returns:
        parser
    """
    parser.add_argument("-w", "--world_size", default=1, type=int)
    parser.add_argument("-p", "--port", default=12345, type=int)
    return parser
