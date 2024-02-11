from dataclasses import dataclass


@dataclass
class BlockDevice:
    """
    Represents a block device.

    Attributes:
        device (str): The device name.
        cluster (str): The cluster name.
        description (str): A description of the block device.
    """

    def __init__(self, device: str, cluster: str, description: str):
        self.device = device
        self.cluster = cluster
        self.description = description

    def __str__(self):
        return (
            f"BlockDevice(device={self.device},"
            f" cluster={self.cluster},"
            f" description={self.description})"
        )
