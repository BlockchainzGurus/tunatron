//! sast-engine: pipeline SAST con call-graph slicing e verdetto del modello locale.

pub mod config;
pub mod graph_slicer;
pub mod ingestor;
pub mod notarizer;
pub mod pipeline;
pub mod prompt;
pub mod rag_cwe;
pub mod server;
pub mod slice_cli;
pub mod vex;
pub mod vllm_client;
