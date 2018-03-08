extern crate image;
extern crate rayon;

use std::fs::File;
use std::io;
use std::path::Path;
use std::f64;
use image::{DynamicImage, open, FilterType, GenericImage, Pixel,RgbaImage,imageops};
use rayon::prelude::*;

type GenError = Box<std::error::Error>;
type GenResult<T> = Result<T, GenError>;
type AverageColor = (DynamicImage,[u8;3]);

/*
** TODO: Prompt user for source image
** Process source image (make a grid based on tile size?)
*/

fn main() {
// Create photomosaic a based on an original image formed from tiles
fn create_mosaic<P>(original: DynamicImage,output: P,tiles: Vec<AverageColor>)
    where P: AsRef<Path>
{
    let mut original = original.clone();
    let (width, height) = original.dimensions();

    let mut new_image = RgbaImage::new(width,height);
    let tile_size = 10;

    let (mut x,mut y) = (0,0);
    while y < height {
        x = 0;
        while x < width {
            let cell = original.crop(x,y,tile_size,tile_size);
            let average_color = get_average_color(&cell);
            let nearest = nearest(&average_color,&tiles);

            new_image.copy_from(nearest,x,y);
            x += tile_size;
        }
        y += tile_size
}

    new_image.save(output).expect("Error saving new_mosaic.png");
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
    let resized_tiles = tiles.par_iter()
        .map(|tile| -> AverageColor {
        let resized = tile.resize(50,50,FilterType::Nearest);
        let average_color = get_average_color(&resized);

            (resized,average_color)
        })
        .collect();
    resized_tiles
}

fn get_average_color(image: &DynamicImage) -> [u8;3] {
    let mut rgb: [usize;3] = [0;3];
    for p in image.pixels() {
        let pix = p.2.to_rgb();
        let channels = pix.channels();

        rgb[0] += channels[0] as usize;
        rgb[1] += channels[1] as usize;
        rgb[2] += channels[2] as usize;
        }
    let p_count = (image.width() * image.height()) as usize;
    for i in 0..rgb.len() {
        rgb[i] = rgb[i] / p_count;
    }

    [rgb[0] as u8, rgb[1] as u8,rgb[2] as u8]
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

// Euclidean distance between 2 RGB color values
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
