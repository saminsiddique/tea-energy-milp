"""Data fetcher modules for NSGA paper replication."""

from data.fetchers.nasa_power import NASAPowerClient
from data.fetchers.saint_martin_data import SaintMartinDataProvider

__all__ = [
    "NASAPowerClient",
    "SaintMartinDataProvider",
]
