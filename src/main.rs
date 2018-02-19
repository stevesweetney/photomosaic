extern crate image;

use std::fs;
use std::io;
use std::path::Path;
use image::{DynamicImage, open, ImageError};
use image::FilterType;

type GenError = Box<std::error::Error>;
type GenResult<T> = Result<T, GenError>;

fn main() {
    println!("Hello, world!");
}

fn load_tiles() -> GenResult<Vec<DynamicImage>> {
    let tiles_path = Path::new("tiles/");
    let mut tiles = Vec::new();

    for entry_result in tiles_path.read_dir()? {
        let entry = entry_result?;
        tiles.push(open(entry.path())?);
    }

    Ok(tiles)
}

fn resize(tiles: Vec<DynamicImage>) -> Vec<DynamicImage> {
    let mut resized = Vec::new();
    for tile in tiles {
        resized.push(tile.resize(50,50,FilterType::Nearest));
    }

    resized
}
