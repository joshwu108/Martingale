"""
store/base.py — Abstract interfaces for revision store and ledger.

Both the file-based (research) and SQLite (production) backends implement these.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from martingale.revision import Revision


class AbstractRevisionStore(ABC):
    @abstractmethod
    def publish(self, table: dict) -> Revision: ...

    @abstractmethod
    def get(self, digest: str) -> Revision: ...

    @abstractmethod
    def __contains__(self, digest: str) -> bool: ...

    @abstractmethod
    def all_digests(self) -> Iterator[str]: ...


class AbstractLedger(ABC):
    @abstractmethod
    def append_trajectory(self, actor_id: int, episode_id: int, records: list): ...

    @abstractmethod
    def load_trajectory(self, actor_id: int, episode_id: int): ...

    @abstractmethod
    def all_trajectories(self): ...
