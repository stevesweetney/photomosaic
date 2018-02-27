extern crate image;

use std::fs;
use std::io;
use std::path::Path;
use image::{DynamicImage, open, FilterType, GenericImage, Pixel};

type GenError = Box<std::error::Error>;
type GenResult<T> = Result<T, GenError>;
type AverageColor = (DynamicImage,u8,u8,u8);

/*
** TODO: Prompt user for source image
** Process source image (make a grid based on tile size?)
*/

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

fn resize(tiles: Vec<DynamicImage>) -> Vec<AverageColor> {
    let mut resized_tiles = Vec::new();
    for tile in tiles {
        let resized = tile.resize(50,50,FilterType::Nearest);

        let (mut r,mut g, mut b): (usize,usize,usize) = (0,0,0);
        for p in resized.pixels() {
            let channels = p.2.channels();
            
            r += channels[0] as usize;
            g += channels[1] as usize;
            b += channels[2] as usize;
        }
        let p_count = resized.pixels().count();
        let (r,g,b) = (r/p_count,g/p_count,b/p_count);

        resized_tiles.push((resized,r as u8,g as u8,b as u8));
    }

    resized_tiles
}
