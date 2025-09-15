use pyo3::prelude::*;
use tokenizers::Tokenizer;

// TODO query and document prefixes have to be set correctly, not with params (builder.rs)
// TODO We need to separate again query max length and document max length

// Custom Python exception for tokenization errors
pyo3::create_exception!(
    tokenization_estimator,
    TokenizationError,
    pyo3::exceptions::PyException
);

#[pyclass]
pub struct TokenizationEstimator {
    tokenizer: Tokenizer,
    query_prefix: String,
    query_length: usize,
    document_prefix: String,
    document_length: usize,
    mask_token_id: u32,
    mask_token: String,
    // Model architecture parameters from config.json
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_dim: usize,
    final_dim: usize, // ColBERT output dimension
}

#[pymethods]
impl TokenizationEstimator {
    /// Creates a new `TokenizationEstimator` instance by loading a tokenizer
    /// from the Hugging Face Hub.
    #[staticmethod]
    #[pyo3(signature = (
        repo_id,
        query_length=None,
        document_length=None,
        query_prefix=None,
        document_prefix=None,
        mask_token=None
    ))]
    pub fn from_pretrained(
        repo_id: &str,
        query_length: Option<usize>,
        document_length: Option<usize>,
        query_prefix: Option<String>,
        document_prefix: Option<String>,
        mask_token: Option<String>,
    ) -> PyResult<Self> {
        // Download tokenizer from Hugging Face Hub
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            TokenizationError::new_err(format!("Failed to initialize HF API: {}", e))
        })?;

        let repo = api.model(repo_id.to_string());
        let tokenizer_filename = repo.get("tokenizer.json").map_err(|e| {
            TokenizationError::new_err(format!(
                "Failed to download tokenizer from {}: {}",
                repo_id, e
            ))
        })?;

        // Download and parse config.json to get model architecture parameters
        let config_filename = repo.get("config.json").map_err(|e| {
            TokenizationError::new_err(format!("Failed to download config from {}: {}", repo_id, e))
        })?;

        let special_tokens_map_filename = repo.get("special_tokens_map.json").map_err(|e| {
            TokenizationError::new_err(format!(
                "Failed to download special_tokens_map from {}: {}",
                repo_id, e
            ))
        })?;

        // Read config file
        let config_content = std::fs::read_to_string(&config_filename).map_err(|e| {
            TokenizationError::new_err(format!("Failed to read config file: {}", e))
        })?;

        // Read special_tokens_map file
        let special_tokens_map_content = std::fs::read_to_string(&special_tokens_map_filename)
            .map_err(|e| {
                TokenizationError::new_err(format!("Failed to read special_tokens_map file: {}", e))
            })?;

        // Parse config JSON
        let config: serde_json::Value = serde_json::from_str(&config_content).map_err(|e| {
            TokenizationError::new_err(format!("Failed to parse config JSON: {}", e))
        })?;

        // Parse config JSON
        let special_tokens_map: serde_json::Value =
            serde_json::from_str(&special_tokens_map_content).map_err(|e| {
                TokenizationError::new_err(format!(
                    "Failed to parse special_tokens_map_content JSON: {}",
                    e
                ))
            })?;

        // Extract model architecture parameters with defaults for common architectures
        let hidden_dim = config["hidden_size"].as_u64().unwrap_or(768) as usize;

        let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(22) as usize;

        let num_heads = config["num_attention_heads"].as_u64().unwrap_or(12) as usize;

        let final_query_prefix = query_prefix
            .unwrap_or_else(|| config["query_prefix"].as_str().unwrap_or("[Q]").to_string());

        let final_document_prefix = document_prefix.unwrap_or_else(|| {
            config["document_prefix"]
                .as_str()
                .unwrap_or("[D]")
                .to_string()
        });

        let final_mask_token = mask_token.unwrap_or_else(|| {
            special_tokens_map["mask_token"]
                .as_str()
                .unwrap_or("[MASK]")
                .to_string()
        });

        // ModernBERT uses "intermediate_size", BERT uses "intermediate_size" too
        // Default to 4x hidden_dim if not specified (768)
        let intermediate_dim = config["intermediate_size"]
            .as_u64()
            .map(|x| x as usize)
            .unwrap_or(hidden_dim * 4);

        // Load tokenizer from downloaded file
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| TokenizationError::new_err(format!("Failed to load tokenizer: {}", e)))?;

        let mask_token_id = tokenizer
            .token_to_id(final_mask_token.as_str())
            .ok_or_else(|| {
                TokenizationError::new_err(format!(
                    "Token '{}' not found in the tokenizer's vocabulary.",
                    final_mask_token
                ))
            })?;

        Ok(Self {
            tokenizer,
            query_prefix: final_query_prefix,
            query_length: query_length.unwrap_or(256),
            document_prefix: final_document_prefix,
            document_length: document_length.unwrap_or(8192),
            mask_token_id,
            mask_token: final_mask_token,
            hidden_dim,
            num_layers,
            num_heads,
            intermediate_dim,
            final_dim: 128, // ColBERT always outputs 128 dimensions
        })
    }

    /// Estimate the dimensions that would result from tokenization
    /// Returns (batch_size, sequence_length)
    pub fn estimate_dimensions(
        &mut self,
        texts: Vec<String>,
        is_query: bool,
    ) -> PyResult<(usize, usize)> {
        if texts.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Input texts cannot be empty",
            ));
        }

        let (prefix, max_length) = if is_query {
            (self.query_prefix.as_str(), self.query_length)
        } else {
            (self.document_prefix.as_str(), self.document_length)
        };

        // Add prefixes
        let texts_with_prefix: Vec<String> = texts
            .iter()
            .map(|text| format!("{}{}", prefix, text))
            .collect();

        // Configure tokenizer for truncation
        let _ = self
            .tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length,
                ..Default::default()
            }));

        // Configure padding based on query vs document
        let padding_params = if is_query {
            // For ColBERT queries, pad to a fixed length with the [MASK] token.
            tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(max_length),
                pad_id: self.mask_token_id,
                pad_token: self.mask_token.clone(),
                ..Default::default()
            }
        } else {
            // Documents are padded to the longest sequence in the batch.
            tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }
        };
        self.tokenizer.with_padding(Some(padding_params));

        // Tokenize the batch
        let encodings = self
            .tokenizer
            .encode_batch(texts_with_prefix, true)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Tokenization failed: {}", e))
            })?;

        let batch_size = encodings.len();
        let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

        Ok((batch_size, seq_len))
    }

    /// Estimate memory usage in bytes
    /// Returns (token_memory, intermediate_memory, total_estimated_memory)
    #[pyo3(signature = (
        texts,
        is_query,
        overhead_multiplier=3.0
    ))]
    pub fn estimate_memory_usage(
        &mut self,
        texts: Vec<String>,
        is_query: bool,
        overhead_multiplier: Option<f64>,
    ) -> PyResult<(usize, usize, usize)> {
        let (batch_size, seq_len) = self.estimate_dimensions(texts, is_query)?;

        // Use model architecture parameters from config
        let hidden_dim = self.hidden_dim;
        let _num_layers = self.num_layers;
        let intermediate_dim = self.intermediate_dim;
        let embedding_dim = self.final_dim;
        let num_heads = self.num_heads;

        // For mixed precision: some operations in fp16 (2 bytes), attention scores in fp32 (4 bytes)
        let fp16_bytes = 2;
        let fp32_bytes = 4;

        // Memory for input tensors (input_ids, attention_mask, token_type_ids, position_ids)
        let token_memory = batch_size * seq_len * 4 * 4; // 4 tensors, int32

        // Persistent memory during forward pass
        // Embeddings (token + position, in fp16)
        let embedding_memory = batch_size * seq_len * hidden_dim * fp16_bytes * 2;

        // Hidden state buffers (current + residual for skip connections, in fp16)
        let hidden_buffers = batch_size * seq_len * hidden_dim * fp16_bytes * 2;

        // Peak memory during a single layer computation (not accumulated across layers!)
        // We calculate the worst case: global attention layer

        // Q, K, V projections (fp16)
        let qkv_memory = batch_size * seq_len * hidden_dim * fp16_bytes * 3;

        // Attention scores and probabilities (must be fp32 for numerical stability)
        let attention_scores = batch_size * num_heads * seq_len * seq_len * fp32_bytes;
        let attention_probs = batch_size * num_heads * seq_len * seq_len * fp32_bytes;

        // Attention output (fp16)
        let attention_output = batch_size * seq_len * hidden_dim * fp16_bytes;

        // FFN computation (fp16)
        let ffn_intermediate = batch_size * seq_len * intermediate_dim * fp16_bytes;
        let ffn_output = batch_size * seq_len * hidden_dim * fp16_bytes;

        // Peak layer memory (worst case: global attention)
        let peak_layer_memory = qkv_memory
            + attention_scores
            + attention_probs
            + attention_output
            + ffn_intermediate
            + ffn_output;

        // Final ColBERT projection (fp16)
        let projection_memory = batch_size * seq_len * embedding_dim * fp16_bytes;

        // Total peak memory during inference
        let intermediate_memory =
            embedding_memory + hidden_buffers + peak_layer_memory + projection_memory;

        // Total with PyTorch overhead (memory fragmentation, temporary buffers)
        // The 3.0 number was chosen by empirical trials. 250 documents with 1000 random chars cause OOM, while 200 documents don't
        let overhead_multiplier = overhead_multiplier.unwrap_or(3.0);
        let total_memory =
            ((token_memory + intermediate_memory) as f64 * overhead_multiplier) as usize;

        Ok((token_memory, intermediate_memory, total_memory))
    }

    /// Check if a batch can fit in available memory
    #[pyo3(signature = (
        texts,
        is_query,
        available_bytes,
        overhead_multiplier=3.0
    ))]
    pub fn can_fit_in_memory(
        &mut self,
        texts: Vec<String>,
        is_query: bool,
        available_bytes: usize,
        overhead_multiplier: Option<f64>,
    ) -> PyResult<bool> {
        let (_, _, estimated_memory) =
            self.estimate_memory_usage(texts, is_query, overhead_multiplier)?;
        Ok(estimated_memory <= available_bytes)
    }

    /// Split texts into optimal batches based on memory constraints
    /// Uses recursive power-of-2 splitting strategy
    #[pyo3(signature = (
        texts,
        is_query,
        available_bytes,
        overhead_multiplier=3.0
    ))]
    pub fn split_into_batches(
        &mut self,
        texts: Vec<String>,
        is_query: bool,
        available_bytes: usize,
        overhead_multiplier: Option<f64>,
    ) -> PyResult<Vec<Vec<String>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Check if all texts fit in memory as a single batch
        if self.can_fit_in_memory(
            texts.clone(),
            is_query,
            available_bytes,
            overhead_multiplier,
        )? {
            return Ok(vec![texts]);
        }

        // Recursively split the texts
        self.split_recursive(texts, is_query, available_bytes, overhead_multiplier)
    }

    /// Recursively split texts into batches that fit in memory
    fn split_recursive(
        &mut self,
        texts: Vec<String>,
        is_query: bool,
        available_bytes: usize,
        overhead_multiplier: Option<f64>,
    ) -> PyResult<Vec<Vec<String>>> {
        // Base case: if we have only one text and it doesn't fit, return error
        if texts.len() == 1 {
            if !self.can_fit_in_memory(
                texts.clone(),
                is_query,
                available_bytes,
                overhead_multiplier,
            )? {
                return Err(TokenizationError::new_err(
                    "Single text is too large to fit in available memory",
                ));
            }
            return Ok(vec![texts]);
        }

        // Check if current batch fits in memory
        if self.can_fit_in_memory(
            texts.clone(),
            is_query,
            available_bytes,
            overhead_multiplier,
        )? {
            return Ok(vec![texts]);
        }

        // Split in half (power of 2 strategy)
        let mid = texts.len() / 2;
        let (left_batch, right_batch) = texts.split_at(mid);

        // Recursively split both halves
        let mut result = Vec::new();

        let left_batches = self.split_recursive(
            left_batch.to_vec(),
            is_query,
            available_bytes,
            overhead_multiplier,
        )?;
        result.extend(left_batches);

        let right_batches = self.split_recursive(
            right_batch.to_vec(),
            is_query,
            available_bytes,
            overhead_multiplier,
        )?;
        result.extend(right_batches);

        Ok(result)
    }
}

/// Python module
#[pymodule]
fn tokenizator(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TokenizationEstimator>()?;
    m.add("TokenizationError", _py.get_type::<TokenizationError>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Skip by default - requires internet connection
    fn test_from_pretrained() {
        let estimator = TokenizationEstimator::from_pretrained(
            "lightonai/GTE-ModernColBERT-v1",
            Some(256),
            Some(8192),
            Some("[Q]".to_string()),
            Some("[D]".to_string()),
            Some("[MASK]".to_string()),
        );
        assert!(estimator.is_ok());
    }
}
