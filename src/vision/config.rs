use candle_nn::Activation;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    pub hidden_act: Activation,
    pub intermediate_size: usize,
    pub num_heads: usize,
    #[serde(alias = "in_channels")]
    pub in_chans: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}
