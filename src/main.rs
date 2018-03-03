extern crate image;

use std::fs;
use std::io;
use std::path::Path;
use std::f64;
use image::{DynamicImage, open, FilterType, GenericImage, Pixel};

type GenError = Box<std::error::Error>;
type GenResult<T> = Result<T, GenError>;
type AverageColor = (DynamicImage,[u8;3]);

/*
** TODO: Prompt user for source image
** Process source image (make a grid based on tile size?)
*/

fn main() {
    println!("Hello, world!");
}


// Load images to be used as tiles in a mosaic
fn load_tiles() -> GenResult<Vec<DynamicImage>> {
    let tiles_path = Path::new("tiles/");
    let mut tiles = Vec::new();

    for entry_result in tiles_path.read_dir()? {
        let entry = entry_result?;
        tiles.push(open(entry.path())?);
    }

    Ok(tiles)
}

// Resize and calculate the average color of each image
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
        let p_count = (resized.width() * resized.height()) as usize;
        let (r,g,b) = (r/p_count,g/p_count,b/p_count);

        resized_tiles.push((resized,[r as u8,g as u8,b as u8]));
    }

    resized_tiles
}

// Select the tile that is the closest match to out target RGB color value
fn nearest<'t>(target: &[u8], tiles: &'t Vec<AverageColor>) -> &'t DynamicImage {
    let mut nearest_tile = &tiles[0].0;
    let mut smallest_dist = f64::MAX;

    for tile in tiles {
        let dist = distance(target,&tile.1);
        if dist < smallest_dist { 
            smallest_dist = dist;
            nearest_tile = &tile.0;
        }
    }

    nearest_tile
}

// Euclidean distance between 2 RBG color values
fn distance(p1: &[u8], p2: &[u8]) -> f64 {
    let square = |x| x * x;
    let diff_sum = (square(p1[0] as i32 - p2[0] as i32) 
        + square(p1[1] as i32 - p2[1] as i32) 
        + square(p1[2] as i32 - p2[2] as i32)) as f64;

    diff_sum.sqrt()
}
    let mut smallest_dist = f64::MAX;

    for tile in tiles {
        let dist = distance(target,&tile.1);
        if dist < smallest_dist { 
            smallest_dist = dist;
            nearest_tile = &tile.0;
        }
    }

    nearest_tile
}

// Euclidean distance between 2 RBG color values
fn distance(p1: &[u8], p2: &[u8]) -> f64 {
    let square = |x| x * x;
    let diff_sum = (square(p1[0] as i32 - p2[0] as i32) 
        + square(p1[1] as i32 - p2[1] as i32) 
        + square(p1[2] as i32 - p2[2] as i32)) as f64;

    diff_sum.sqrt()
}

#[cfg(test)]
mod test {
    use super::distance;

    #[test]
    fn test_distance() {
        let black = &[0,0,0];
        let white = &[255,255,255];

        let p = &[121,30,177];
        let q = &[237,22,88];

        assert_eq!(distance(black,white).floor(),441.0);
        assert_eq!(distance(p,q).floor(),146.0);
    }

    #[test]
    fn test_distance_between_same_point() {
        let p1 = &[2,3,5];
        let p2 = &[235,110,75];

        assert_eq!(distance(p1,p1),0.0);
        assert_eq!(distance(p2,p2),0.0);
    }
}
