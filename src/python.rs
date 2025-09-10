use pyo3::prelude::*;
use tokenizers::Tokenizer;

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
    document_prefix: String,
    document_length: usize,
    mask_token_id: u32,
    mask_token: String,
}

#[pymethods]
impl TokenizationEstimator {
    /// Creates a new `TokenizationEstimator` instance by loading a tokenizer
    /// from the Hugging Face Hub.
    #[staticmethod]
    #[pyo3(signature = (
        repo_id,
        document_length=None,
        query_prefix=None,
        document_prefix=None,
        mask_token=None
    ))]
    pub fn from_pretrained(
        repo_id: &str,
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

        // Load tokenizer from downloaded file
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| TokenizationError::new_err(format!("Failed to load tokenizer: {}", e)))?;

        // Get mask token info from tokenizer or use provided mask_token
        let mask_token_str = mask_token.unwrap_or_else(|| "[MASK]".to_string());
        let mask_token_id = tokenizer
            .get_vocab(true)
            .get(&mask_token_str)
            .copied()
            .or_else(|| tokenizer.get_vocab(true).get("[MASK]").copied())
            .or_else(|| tokenizer.get_vocab(true).get("[UNK]").copied())
            .unwrap_or(100); // fallback ID

        Ok(Self {
            tokenizer,
            query_prefix: query_prefix.unwrap_or_else(|| "[Q]".to_string()),
            document_prefix: document_prefix.unwrap_or_else(|| "[D]".to_string()),
            document_length: document_length.unwrap_or(8192),
            mask_token_id,
            mask_token: mask_token_str,
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
            (self.query_prefix.as_str(), self.document_length)
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
    pub fn estimate_memory_usage(
        &mut self,
        texts: Vec<String>,
        is_query: bool,
        embedding_dim: Option<usize>,
        bytes_per_token: Option<usize>,
    ) -> PyResult<(usize, usize, usize)> {
        let (batch_size, seq_len) = self.estimate_dimensions(texts, is_query)?;
        // 128 dimensions usually in late interaction models
        let embedding_dim = embedding_dim.unwrap_or(128);
        // Assuming 4 bytes per token (float32)
        let bytes_per_token = bytes_per_token.unwrap_or(4);

        // Memory for token tensors (input_ids, attention_mask, token_type_ids)
        // token_type_ids is usually set to 0 for all tokens, is a legacy param for next sentence prediction
        let token_memory = batch_size * seq_len * 3 * bytes_per_token;

        // Memory for intermediate embeddings and final output
        let intermediate_memory = batch_size * seq_len * embedding_dim * bytes_per_token * 2; // intermediate + final layers

        // Add 20% overhead for temporary tensors and operations
        let total_memory = ((token_memory + intermediate_memory) as f64 * 1.2) as usize;

        Ok((token_memory, intermediate_memory, total_memory))
    }

    /// Check if a batch can fit in available memory
    pub fn can_fit_in_memory(
        &mut self,
        texts: Vec<String>,
        is_query: bool,
        available_bytes: usize,
        embedding_dim: Option<usize>,
        bytes_per_token: Option<usize>,
    ) -> PyResult<bool> {
        let (_, _, estimated_memory) =
            self.estimate_memory_usage(texts, is_query, embedding_dim, bytes_per_token)?;
        Ok(estimated_memory <= available_bytes)
    }

    /// Get optimal batch size for available memory
    pub fn get_optimal_batch_size(
        &mut self,
        sample_text: String,
        is_query: bool,
        available_bytes: usize,
        embedding_dim: Option<usize>,
        bytes_per_token: Option<usize>,
    ) -> PyResult<usize> {
        // Start with a single text to get per-item memory usage
        let single_text = vec![sample_text.clone()];
        let (_, _, memory_per_item) =
            self.estimate_memory_usage(single_text, is_query, embedding_dim, bytes_per_token)?;

        if memory_per_item == 0 {
            return Ok(1);
        }

        // Calculate max batch size (with some safety margin)
        let max_batch_size = (available_bytes as f64 * 0.8) as usize / memory_per_item;
        Ok(max_batch_size.max(1))
    }

    /// Split texts into optimal batches based on memory constraints
    pub fn split_into_batches(
        &mut self,
        texts: Vec<String>,
        is_query: bool,
        available_bytes: usize,
        embedding_dim: Option<usize>,
        bytes_per_token: Option<usize>,
    ) -> PyResult<Vec<Vec<String>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let optimal_batch_size = self.get_optimal_batch_size(
            texts[0].clone(),
            is_query,
            available_bytes,
            embedding_dim,
            bytes_per_token,
        )?;

        let mut batches = Vec::new();
        for chunk in texts.chunks(optimal_batch_size) {
            batches.push(chunk.to_vec());
        }

        Ok(batches)
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
            Some(8192),
            Some("[Q]".to_string()),
            Some("[D]".to_string()),
            Some("[MASK]".to_string()),
        );
        assert!(estimator.is_ok());
    }
}
