//! HNSW (Hierarchical Navigable Small World) index for O(log n) approximate
//! nearest-neighbor search. Used by PredaCore's memory store at scale when
//! the linear SIMD scan becomes too slow (typically above ~10k vectors).
//!
//! API surface (exposed as `predacore_core.PyHnswIndex`):
//!   - `new(dims, max_nb_connection, ef_construction, max_elements)` — constructor
//!   - `insert(id, vector)` — add a vector with its memory id, O(log n)
//!   - `search(query, top_k, ef_search)` — return [(id, similarity)], O(log n)
//!   - `len()`, `dims()` — basic introspection
//!
//! Default parameters tuned for 384-dim BGE embeddings + PredaCore memory
//! scale (most users have <100k rows):
//!   M = 16          (max edges per node)
//!   ef_construction = 200  (index build quality)
//!   ef_search = 50  (query search width; higher = better recall)
//!
//! Distance: cosine. Returns similarity (1 − cosine_distance) for callers.
//! Thread-safety: all mutation guarded by an internal Mutex so Python
//! callers can use it from multiple asyncio tasks safely.
//!
//! NOT YET IMPLEMENTED: save/load to disk. For now the Python layer
//! rebuilds the index from SQLite on daemon startup. Adding file_dump /
//! file_load is a straightforward extension using hnsw_rs::hnswio, tracked
//! as a follow-up task.

use hnsw_rs::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Mutex;

/// Python-visible HNSW index. Wraps an hnsw_rs::Hnsw internally with
/// a parallel id_map tracking which memory ID lives at which HNSW slot.
#[pyclass]
pub struct PyHnswIndex {
    inner: Mutex<HnswInner>,
}

struct HnswInner {
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// HNSW internal position → memory ID
    id_map: Vec<String>,
    dims: usize,
}

#[pymethods]
impl PyHnswIndex {
    #[new]
    #[pyo3(signature = (dims, max_nb_connection = 16, ef_construction = 200, max_elements = 1_000_000))]
    fn new(
        dims: usize,
        max_nb_connection: usize,
        ef_construction: usize,
        max_elements: usize,
    ) -> PyResult<Self> {
        // max_layer = 16 is the hnsw_rs recommended value for most workloads.
        let max_layer = 16;
        let hnsw = Hnsw::<f32, DistCosine>::new(
            max_nb_connection,
            max_elements,
            max_layer,
            ef_construction,
            DistCosine {},
        );
        Ok(PyHnswIndex {
            inner: Mutex::new(HnswInner {
                hnsw,
                id_map: Vec::new(),
                dims,
            }),
        })
    }

    /// Insert a single vector with its memory ID. Returns the internal slot.
    fn insert(&self, id: String, vector: Vec<f32>) -> PyResult<usize> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("HNSW lock poisoned"))?;
        if vector.len() != guard.dims {
            return Err(PyRuntimeError::new_err(format!(
                "vector dim {} != index dim {}",
                vector.len(),
                guard.dims
            )));
        }
        let pos = guard.id_map.len();
        guard.id_map.push(id);
        // hnsw_rs takes (data_slice, DataId). We use our position as the DataId
        // so results map 1:1 back to id_map without extra bookkeeping.
        guard.hnsw.insert((&vector, pos));
        Ok(pos)
    }

    /// Search top-k most similar vectors. Returns (id, similarity) pairs
    /// sorted by similarity descending. ``ef_search`` controls recall
    /// quality — higher = more accurate but slower. At small N, set
    /// ef_search close to N for near-exact recall.
    #[pyo3(signature = (query, top_k = 10, ef_search = 50))]
    fn search(
        &self,
        query: Vec<f32>,
        top_k: usize,
        ef_search: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("HNSW lock poisoned"))?;
        if query.len() != guard.dims {
            return Err(PyRuntimeError::new_err(format!(
                "query dim {} != index dim {}",
                query.len(),
                guard.dims
            )));
        }
        let neighbours = guard.hnsw.search(&query, top_k, ef_search);
        // DistCosine returns distance in [0, 2]. For our caller's purposes
        // we want "similarity" where 1.0 = identical. Convert via (1 - dist).
        // Clamp to [-1, 1] range for safety against float weirdness.
        Ok(neighbours
            .into_iter()
            .filter_map(|n| {
                guard.id_map.get(n.d_id).map(|id| {
                    let sim = (1.0f32 - n.distance).clamp(-1.0, 1.0);
                    (id.clone(), sim)
                })
            })
            .collect())
    }

    /// Number of vectors currently indexed.
    fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.id_map.len())
            .unwrap_or(0)
    }

    /// Embedding dimension this index was built for.
    fn dims(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.dims)
            .unwrap_or(0)
    }

    /// Debug repr showing size + params.
    fn __repr__(&self) -> String {
        match self.inner.lock() {
            Ok(g) => format!("PyHnswIndex(dims={}, len={})", g.dims, g.id_map.len()),
            Err(_) => "PyHnswIndex(<locked>)".into(),
        }
    }
}
