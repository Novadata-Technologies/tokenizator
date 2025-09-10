// Main library entry point for tokenizator
// Handles feature-based compilation for Python bindings

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python")]
pub use python::*;

// For non-Python builds, we can expose core functionality here
#[cfg(not(feature = "python"))]
pub mod core {
    // Core tokenization logic without Python bindings
    // This allows the library to be used in pure Rust contexts
    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

// Re-export core functionality for non-Python builds
#[cfg(not(feature = "python"))]
pub use core::*;
