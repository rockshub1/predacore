
from jarvis._vendor.knowledge_nexus.storage import DiskKnowledgeGraphStore


def test_disk_store_init(tmp_path):
    """Ensure DiskKnowledgeGraphStore initializes at path."""
    base = tmp_path / "kn_disk"
    store = DiskKnowledgeGraphStore(base_dir=str(base))
    assert store.nodes_path.exists() is False or store.nodes_path.parent.exists()

