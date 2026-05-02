/// BGE-small-en-v1.5 embedding via Candle — no PyTorch dependency.
///
/// Model: BAAI/bge-small-en-v1.5 (384-dim, BERT-based, MTEB 62.2)
/// Maintained by BAAI (Beijing Academy of AI). Uses mean pooling + L2 norm.
/// Downloads from HuggingFace Hub on first use, cached at ~/.cache/huggingface/
/// (~133 MB, one-time download).

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::Mutex;
use tokenizers::Tokenizer;

use once_cell::sync::Lazy;

const MODEL_ID: &str = "BAAI/bge-small-en-v1.5";
// Phase 6b: raised 256 → 512 to match lab's kernel. Auto-chunking keeps
// each chunk ≤ ~200 tokens either way, but longer un-chunked memories
// (200-512 tokens) now get their full content embedded in one pass
// instead of truncated.
const MAX_SEQ_LEN: usize = 512;

/// Cached model + tokenizer (loaded once, reused across calls).
static MODEL: Lazy<Mutex<Option<EmbeddingModel>>> = Lazy::new(|| Mutex::new(None));

struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

/// Load the GTE-small model from HuggingFace Hub (or cache).
fn load_model() -> Result<EmbeddingModel, String> {
    let device = Device::Cpu;

    // Download model files via hf-hub
    let api = Api::new().map_err(|e| format!("HF Hub API error: {e}"))?;
    let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| format!("Failed to download tokenizer: {e}"))?;
    let config_path = repo
        .get("config.json")
        .map_err(|e| format!("Failed to download config: {e}"))?;
    let weights_path = repo
        .get("model.safetensors")
        .map_err(|e| format!("Failed to download weights: {e}"))?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    // Load config
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read config: {e}"))?;
    let config: BertConfig =
        serde_json::from_str(&config_str).map_err(|e| format!("Failed to parse config: {e}"))?;

    // Load weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)
            .map_err(|e| format!("Failed to load weights: {e}"))?
    };

    let model =
        BertModel::load(vb, &config).map_err(|e| format!("Failed to build model: {e}"))?;

    Ok(EmbeddingModel {
        model,
        tokenizer,
        device,
    })
}

/// Ensure the model is loaded (lazy init).
fn ensure_model() -> Result<(), String> {
    let mut guard = MODEL.lock().map_err(|e| format!("Lock error: {e}"))?;
    if guard.is_none() {
        *guard = Some(load_model()?);
    }
    Ok(())
}

/// Maximum batch size for a single forward pass. Bounded so output tensor
/// stays under ~50 MB (BATCH × MAX_SEQ_LEN × hidden=384 × 4 bytes ≈ 25 MB
/// at 32 batch). Larger batches give marginal speedup beyond ~32 because
/// Candle's CPU forward pass already multi-threads internally.
const EMBED_BATCH_SIZE: usize = 32;

/// Embed a batch of texts using BGE-small with REAL batched forward.
///
/// Optimization story (PR T5c):
///   - **Parallel tokenization** via rayon: tokenization is CPU-bound and
///     embarrassingly parallel. par_iter() over texts gets us linear
///     speedup across cores during preprocessing.
///   - **Real batched forward pass**: previous version called the model
///     once per text (batch_size=1), wasting ~5× the per-call setup.
///     We now stack texts into a single (batch, max_len) tensor padded
///     with attention_mask=0 and call forward once per batch.
///   - **Bounded batch size**: caps memory at ~25 MB per batch so we
///     don't OOM on long input lists. Texts beyond the limit chunk
///     transparently and outputs are concatenated in input order.
///
/// API stability: ``embed(texts: &[String]) -> Vec<Vec<f32>>`` is
/// unchanged — Python callers see a faster function with the same shape.
pub fn embed(texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
    use rayon::prelude::*;

    if texts.is_empty() {
        return Ok(Vec::new());
    }

    ensure_model()?;
    let guard = MODEL.lock().map_err(|e| format!("Lock error: {e}"))?;
    let m = guard.as_ref().ok_or("Model not loaded")?;

    // ── Phase 1: parallel tokenization ──────────────────────────────────
    // tokenizers::Tokenizer is internally Send+Sync for its core path; we
    // borrow `&m.tokenizer` from each rayon worker. Truncation + tensor-
    // building stays sequential (cheap copies; the heavy work is the
    // model forward pass).
    let encodings: Vec<_> = texts
        .par_iter()
        .map(|t| {
            m.tokenizer
                .encode(t.as_str(), true)
                .map_err(|e| format!("Tokenization error: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

    // ── Phase 2: batched forward, chunked to bound memory ───────────────
    for chunk_encodings in encodings.chunks(EMBED_BATCH_SIZE) {
        let batch_size = chunk_encodings.len();

        // Truncate per-row + find batch-local max length.
        let mut row_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut row_type_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut row_masks: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut max_len = 0usize;
        for enc in chunk_encodings {
            let mut ids = enc.get_ids().to_vec();
            let mut tids = enc.get_type_ids().to_vec();
            let mut mask = enc.get_attention_mask().to_vec();
            if ids.len() > MAX_SEQ_LEN {
                ids.truncate(MAX_SEQ_LEN);
                tids.truncate(MAX_SEQ_LEN);
                mask.truncate(MAX_SEQ_LEN);
            }
            max_len = max_len.max(ids.len());
            row_ids.push(ids);
            row_type_ids.push(tids);
            row_masks.push(mask);
        }
        // Guard against degenerate empty-tokenization (would yield 0×0 tensor).
        if max_len == 0 {
            for _ in 0..batch_size {
                all_embeddings.push(vec![0.0_f32; embedding_dim()]);
            }
            continue;
        }

        // Pad to batch-local max with token_id=0 (BERT [PAD] convention) +
        // attention_mask=0 so the model ignores padded positions during
        // attention AND mean pooling.
        for ((ids, tids), mask) in row_ids
            .iter_mut()
            .zip(row_type_ids.iter_mut())
            .zip(row_masks.iter_mut())
        {
            while ids.len() < max_len {
                ids.push(0);
                tids.push(0);
                mask.push(0);
            }
        }

        // Flatten to (batch * max_len) for tensor construction.
        let flat_ids: Vec<u32> = row_ids.into_iter().flatten().collect();
        let flat_tids: Vec<u32> = row_type_ids.into_iter().flatten().collect();
        let flat_mask: Vec<u32> = row_masks.into_iter().flatten().collect();

        let input_ids = Tensor::new(flat_ids.as_slice(), &m.device)
            .map_err(|e| format!("input_ids tensor: {e}"))?
            .reshape((batch_size, max_len))
            .map_err(|e| format!("input_ids reshape: {e}"))?;
        let token_type_ids = Tensor::new(flat_tids.as_slice(), &m.device)
            .map_err(|e| format!("token_type_ids tensor: {e}"))?
            .reshape((batch_size, max_len))
            .map_err(|e| format!("token_type_ids reshape: {e}"))?;
        let attention_mask_tensor = Tensor::new(flat_mask.as_slice(), &m.device)
            .map_err(|e| format!("attention_mask tensor: {e}"))?
            .reshape((batch_size, max_len))
            .map_err(|e| format!("attention_mask reshape: {e}"))?;

        // Single forward pass over the whole batch.
        let output = m
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask_tensor))
            .map_err(|e| format!("Forward pass error: {e}"))?;

        // Mean pooling with attention mask broadcast over the hidden dim.
        let mask_f32 = attention_mask_tensor
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| format!("Mask dtype: {e}"))?
            .unsqueeze(2)
            .map_err(|e| format!("Mask unsqueeze: {e}"))?;
        let masked = output
            .broadcast_mul(&mask_f32)
            .map_err(|e| format!("Masked mul: {e}"))?;
        let summed = masked.sum(1).map_err(|e| format!("Sum: {e}"))?;
        let mask_sum = mask_f32
            .sum(1)
            .map_err(|e| format!("Mask sum: {e}"))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| format!("Mask clamp: {e}"))?;
        let mean_pooled = summed
            .broadcast_div(&mask_sum)
            .map_err(|e| format!("Mean pool: {e}"))?;

        // L2 normalize per row → (batch, dim).
        let norm = mean_pooled
            .sqr()
            .map_err(|e| format!("Sqr: {e}"))?
            .sum(1)
            .map_err(|e| format!("Norm sum: {e}"))?
            .sqrt()
            .map_err(|e| format!("Sqrt: {e}"))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| format!("Norm clamp: {e}"))?
            .unsqueeze(1)
            .map_err(|e| format!("Norm unsqueeze: {e}"))?;
        let normalized = mean_pooled
            .broadcast_div(&norm)
            .map_err(|e| format!("Normalize: {e}"))?;

        // Extract per-row Vec<f32>. (batch, dim) → flat → split.
        let dim = embedding_dim();
        let flat: Vec<f32> = normalized
            .flatten_all()
            .map_err(|e| format!("Flatten: {e}"))?
            .to_vec1()
            .map_err(|e| format!("ToVec: {e}"))?;
        if flat.len() != batch_size * dim {
            return Err(format!(
                "shape mismatch: got {} floats, expected {} × {} = {}",
                flat.len(), batch_size, dim, batch_size * dim,
            ));
        }
        for i in 0..batch_size {
            all_embeddings.push(flat[i * dim..(i + 1) * dim].to_vec());
        }
    }

    Ok(all_embeddings)
}

/// Get the embedding dimension (384 for GTE-small).
pub fn embedding_dim() -> usize {
    384
}

/// Check if the model is already loaded/cached.
pub fn is_model_loaded() -> bool {
    MODEL
        .lock()
        .map(|g| g.is_some())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dim() {
        assert_eq!(embedding_dim(), 384);
    }

    #[test]
    fn test_embed_empty() {
        let result = embed(&[]).unwrap();
        assert!(result.is_empty());
    }

    // Note: Full embedding tests require model download (~67MB)
    // Run with: cargo test -- --ignored
    #[test]
    #[ignore]
    fn test_embed_single() {
        let texts = vec!["Hello world".to_string()];
        let result = embed(&texts).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);
        // Check L2 normalization (norm should be ~1.0)
        let norm: f32 = result[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "norm = {norm}");
    }

    #[test]
    #[ignore]
    fn test_embed_batch() {
        let texts = vec![
            "Python is great for AI".to_string(),
            "Rust is fast and safe".to_string(),
        ];
        let result = embed(&texts).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 384);
        assert_eq!(result[1].len(), 384);
    }

    #[test]
    #[ignore]
    fn test_embed_similarity() {
        let texts = vec![
            "Python programming language".to_string(),
            "Rust programming language".to_string(),
            "Banana smoothie recipe".to_string(),
        ];
        let result = embed(&texts).unwrap();
        // Python and Rust should be more similar to each other than to banana
        let sim_py_rust = crate::vector::cosine_similarity(&result[0], &result[1]);
        let sim_py_banana = crate::vector::cosine_similarity(&result[0], &result[2]);
        assert!(
            sim_py_rust > sim_py_banana,
            "Python-Rust ({sim_py_rust:.3}) should be more similar than Python-Banana ({sim_py_banana:.3})"
        );
    }
}
