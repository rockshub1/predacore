//! TinyClick — UI element grounding for browser vision (T4c).
//!
//! Status: SKELETON — not wired into lib.rs yet.
//!
//! When activated this module provides a single Python-facing function
//! ``parse_screen(png_bytes, query) -> (x, y, role, label, confidence)``
//! that runs Samsung's TinyClick (Florence-2-base finetune, 0.27B params,
//! MIT) through the ``ort`` crate, with weights downloaded via the
//! same ``hf_hub`` Rust crate that BGE already uses.
//!
//! # Why TinyClick over UGround / OmniParser
//!
//! See ``predacore/operators/vision_model_loader.py`` for the model
//! decision rationale. TL;DR: 0.27B params, 73.8% on ScreenSpot,
//! MIT license, sub-second inference, Florence-2-base backbone with
//! official Microsoft ONNX exports.
//!
//! # Cache layout — match BGE
//!
//! BGE uses ``~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/``
//! via the ``hf_hub`` crate's standard cache resolution. TinyClick will
//! use ``~/.cache/huggingface/hub/models--Samsung--TinyClick/`` — same
//! root directory, same crate, so users have ONE cache to manage.
//!
//! # Activation steps (follow-up PR)
//!
//! 1. Add to ``Cargo.toml`` (``hf_hub`` is already there for BGE):
//!    ```toml
//!    [dependencies]
//!    ort = { version = "2.0", features = ["coreml", "cuda", "load-dynamic"] }
//!    image = "0.25"
//!    ndarray = "0.16"
//!    # hf_hub = "0.3"  # already present for BGE
//!    ```
//!
//! 2. Implement the loader (mirrors ``embedding.rs::load_model``):
//!    ```ignore
//!    use hf_hub::{api::sync::Api, Repo, RepoType};
//!    let api = Api::new()?;
//!    let repo = api.repo(Repo::new("Samsung/TinyClick".into(), RepoType::Model));
//!    let onnx = repo.get("model.onnx")?;          // weights
//!    let tokenizer = repo.get("tokenizer.json")?; // input prep
//!    let session = ort::Session::builder()?
//!        .with_execution_providers([ort::CoreMLExecutionProvider::default()
//!            .with_subgraphs().build()])?
//!        .commit_from_file(onnx)?;
//!    ```
//!
//! 3. Uncomment the ``mod omniparser;`` line in ``lib.rs`` (rename to
//!    ``mod tinyclick;`` for clarity) and register the
//!    ``#[pyfunction] parse_screen`` Python binding.
//!
//! 4. Run ``maturin develop`` — the predacore_core wheel rebuilds with
//!    the new function.
//!
//! # Why Rust + ``ort`` over Python + onnxruntime
//!
//! - CoreML execution provider lights up Apple Silicon's ANE for
//!   ~100-200ms inference (vs ~400ms on CPU Python).
//! - No GIL — preprocessing (image decode + resize) and inference can
//!   parallelize when the bridge runs multiple browser tabs at once.
//! - Lower memory: ~600MB loaded fp16 vs ~1GB Python equivalent.
//! - Reuses our existing Candle / PyO3 / hf_hub stack — no new
//!   dependencies beyond ``ort``.
//!
//! # Output contract
//!
//! TinyClick is a *grounding* model — input is screenshot + textual
//! query ("click the login button"), output is pixel coordinates of
//! the matching element. Returned tuple:
//!   - ``x, y``        — center pixel coords in viewport
//!   - ``role``        — "button" | "link" | "input" | "icon" | "text" | "image"
//!   - ``label``       — element's accessible text (best-effort from model)
//!   - ``confidence``  — 0..1 model score
//!
//! These map 1:1 to ``VisionElement`` on the Python side.

#![allow(dead_code)]

/// One grounded UI element. Mirrors :class:`VisionElement`.
pub struct Element {
    pub index: u32,
    pub bbox: (i32, i32, i32, i32),
    pub role: String,
    pub text: String,
    pub confidence: f32,
}

/// PLACEHOLDER — replace with ort::Session-based inference.
///
/// When this is wired up, the function signature matches what the
/// Python wrapper expects:
///
/// ```ignore
/// #[pyfunction]
/// pub fn parse_screen(png_bytes: &[u8]) -> PyResult<Vec<(i32, i32, i32, i32, String, String, f32)>> {
///     // 1. decode png_bytes -> RGB image (image crate)
///     // 2. resize to 640x480 (model input)
///     // 3. normalize + to NCHW tensor (ndarray)
///     // 4. session.run(input) -> outputs (bbox heatmap + class logits)
///     // 5. decode boxes via NMS, classify role, return tuples
///     Ok(vec![])
/// }
/// ```
///
/// Until the dep + weights are in place this returns an empty vec so
/// the Python side falls back to the cloud-vision provider.
pub fn parse_screen(_png_bytes: &[u8]) -> Vec<Element> {
    Vec::new()
}
