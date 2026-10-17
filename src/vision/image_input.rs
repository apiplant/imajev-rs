use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use image::imageops::FilterType;

const PATCH: u32 = 16;
const MERGE: u32 = 2;
const TEMPORAL: usize = 2;
const MAX_PIXELS: u64 = 390_000;

pub struct PreparedPageImage {
    pub pixel_values: Tensor,
    pub grid_thw: Tensor,
    pub image_tokens: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

fn target_size(width: u32, height: u32) -> (u32, u32) {
    let factor = PATCH * MERGE;
    let min_pixels = 65_536u64;
    let pixels = width as u64 * height as u64;
    let scale = if pixels > MAX_PIXELS {
        (MAX_PIXELS as f64 / pixels as f64).sqrt()
    } else if pixels < min_pixels {
        (min_pixels as f64 / pixels.max(1) as f64).sqrt()
    } else {
        1.0
    };
    let mut w = (((width as f64 * scale / factor as f64).round() as u32).max(1)) * factor;
    let mut h = (((height as f64 * scale / factor as f64).round() as u32).max(1)) * factor;
    if w as u64 * h as u64 > MAX_PIXELS {
        let correction = (MAX_PIXELS as f64 / (w as u64 * h as u64) as f64).sqrt();
        w = (((w as f64 * correction / factor as f64).floor() as u32).max(1)) * factor;
        h = (((h as f64 * correction / factor as f64).floor() as u32).max(1)) * factor;
    }
    (w.max(factor), h.max(factor))
}

pub fn prepare_page_image(bytes: &[u8], device: &Device) -> Result<PreparedPageImage> {
    let decoded = image::load_from_memory(bytes)?;
    let (width, height) = target_size(decoded.width(), decoded.height());
    if width as u64 * height as u64 > MAX_PIXELS {
        bail!("resized page image exceeds the 390,000 pixel Imajev limit");
    }
    let rgb = decoded
        .resize_exact(width, height, FilterType::CatmullRom)
        .to_rgb8();
    let grid_h = height / PATCH;
    let grid_w = width / PATCH;
    let patches = (grid_h * grid_w) as usize;
    let mut values = Vec::with_capacity(patches * 3 * TEMPORAL * (PATCH * PATCH) as usize);

    // Qwen's spatial merger consumes consecutive 2x2 patches as one token. Pack
    // patches in block,inner-row,inner-column order so pixel tokens line up with
    // the vision tower's interpolated positional embeddings and rotary positions.
    for block_y in 0..grid_h / MERGE {
        for block_x in 0..grid_w / MERGE {
            for inner_y in 0..MERGE {
                for inner_x in 0..MERGE {
                    let patch_y = block_y * MERGE + inner_y;
                    let patch_x = block_x * MERGE + inner_x;
                    for channel in 0..3 {
                        for _time in 0..TEMPORAL {
                            for y in 0..PATCH {
                                for x in 0..PATCH {
                                    let pixel =
                                        rgb.get_pixel(patch_x * PATCH + x, patch_y * PATCH + y);
                                    values.push(pixel[channel] as f32 / 127.5 - 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let pixel_values = Tensor::from_vec(values, (patches, 3 * TEMPORAL * 16 * 16), device)?;
    let grid_thw = Tensor::from_vec(vec![1u32, grid_h, grid_w], (1, 3), device)?;
    let merge_unit = (MERGE * MERGE) as usize;
    if patches % merge_unit != 0 {
        bail!("patch grid is not divisible by the configured merge size");
    }
    Ok(PreparedPageImage {
        pixel_values,
        grid_thw,
        image_tokens: patches / merge_unit,
        grid_h: grid_h as usize,
        grid_w: grid_w as usize,
    })
}
