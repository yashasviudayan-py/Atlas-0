"""In-memory semantic label store with confidence-weighted updates.

Maps object identifiers (cluster IDs) to :class:`~atlas.vlm.inference.SemanticLabel`
objects and serialises them to protobuf for IPC with the Rust layer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import structlog

from atlas.vlm.inference import SemanticLabel

logger = structlog.get_logger(__name__)

try:
    from atlas.utils import atlas_pb2  # type: ignore[attr-defined]

    _PROTO_AVAILABLE = True
except ImportError:
    _PROTO_AVAILABLE = False


@dataclass
class LabelEntry:
    """A label entry stored in the :class:`LabelStore`.

    Args:
        object_id: Unique identifier for the object (e.g. DBSCAN cluster ID).
        label: Current best semantic label for this object.
        update_count: Number of times this entry has been written.
    """

    object_id: int
    label: SemanticLabel
    update_count: int = 0


class LabelStore:
    """Thread-safe in-memory store mapping object IDs to semantic labels.

    Updates are confidence-weighted: a new label only replaces an existing one
    when its confidence strictly exceeds the stored value.

    Example::

        store = LabelStore()
        store.update(42, SemanticLabel("glass", "glass", 0.15, 0.9, 0.3, 0.85))
        entry = store.get(42)
        assert entry is not None and entry.label.label == "glass"

        # Lower confidence is rejected.
        store.update(42, SemanticLabel("cup", "ceramic", 0.2, 0.7, 0.4, 0.5))
        assert store.get(42).label.label == "glass"  # unchanged
    """

    def __init__(self) -> None:
        self._entries: dict[int, LabelEntry] = {}
        self._lock = threading.Lock()

    def update(self, object_id: int, new_label: SemanticLabel) -> bool:
        """Insert or replace a label using confidence-weighted selection.

        Args:
            object_id: Integer object identifier.
            new_label: Newly computed :class:`SemanticLabel`.

        Returns:
            ``True`` if the store was updated, ``False`` if the existing entry
            had equal or higher confidence and was retained.
        """
        with self._lock:
            existing = self._entries.get(object_id)
            if existing is not None and existing.label.confidence >= new_label.confidence:
                logger.debug(
                    "label_store_update_skipped",
                    object_id=object_id,
                    stored_conf=existing.label.confidence,
                    new_conf=new_label.confidence,
                )
                return False

            count = (existing.update_count + 1) if existing is not None else 1
            self._entries[object_id] = LabelEntry(
                object_id=object_id,
                label=new_label,
                update_count=count,
            )
            logger.debug(
                "label_store_updated",
                object_id=object_id,
                label=new_label.label,
                confidence=new_label.confidence,
                update_count=count,
            )
            return True

    def get(self, object_id: int) -> LabelEntry | None:
        """Return the stored entry for *object_id*, or ``None``.

        Args:
            object_id: Integer object identifier.

        Returns:
            The :class:`LabelEntry` or ``None`` if not present.
        """
        with self._lock:
            return self._entries.get(object_id)

    def all_entries(self) -> list[LabelEntry]:
        """Return a snapshot of all current label entries.

        Returns:
            List of all :class:`LabelEntry` values (order not guaranteed).
        """
        with self._lock:
            return list(self._entries.values())

    def remove(self, object_id: int) -> bool:
        """Remove the entry for *object_id*.

        Args:
            object_id: Integer object identifier.

        Returns:
            ``True`` if the entry existed and was removed.
        """
        with self._lock:
            return self._entries.pop(object_id, None) is not None

    def clear(self) -> None:
        """Remove all stored labels."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def to_proto_list(self) -> list[object]:
        """Serialise all entries to protobuf ``SemanticLabel`` messages.

        Returns:
            List of ``atlas_pb2.SemanticLabel`` objects ready for IPC.  Returns
            an empty list when the protobuf module is unavailable.
        """
        if not _PROTO_AVAILABLE:
            logger.warning("proto_unavailable_skipping_serialisation")
            return []

        proto_labels = []
        for entry in self.all_entries():
            sl = entry.label
            proto = atlas_pb2.SemanticLabel(  # type: ignore[attr-defined]
                object_id=entry.object_id,
                label=sl.label,
                material=sl.material,
                mass_kg=sl.mass_kg,
                fragility=sl.fragility,
                friction=sl.friction,
                confidence=sl.confidence,
            )
            proto_labels.append(proto)
        return proto_labels
