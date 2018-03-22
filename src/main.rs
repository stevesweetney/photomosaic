extern crate image;
extern crate imageproc;
extern crate rayon;

use std::fs::File;
use std::io;
use std::path::Path;
use std::f64;
use image::{DynamicImage, open, FilterType, GenericImage, Pixel,RgbaImage,GrayImage};
use image::math::utils;
use image::imageops::colorops;
use imageproc::edges;
use rayon::prelude::*;
use rayon::iter::Either;

type GenError = Box<std::error::Error>;
type GenResult<T> = Result<T, GenError>;
enum AverageColor {
    Homogenous { image: DynamicImage, color: [u8;3] },
    Non { image: DynamicImage, color: [u8;3], edges: Option<GrayImage> }
}

impl AverageColor {
    fn get_image(&self) -> &DynamicImage {
        match *self {
            AverageColor::Homogenous { ref image,.. } => image,
            AverageColor::Non { ref image,.. } => image
        }
    }

    fn get_color(&self) -> &[u8] {
        match *self {
            AverageColor::Homogenous { ref color,.. } => color,
            AverageColor::Non { ref color,.. } => color
        }
    }
}

struct RGBHistogram<'im> {
    r_histogram: [u32;256],
    g_histogram: [u32;256],
    b_histogram: [u32;256],
    image: &'im DynamicImage
}

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

// Load and resize images to be used as tiles in a mosaic
fn load_tiles() -> GenResult<Vec<DynamicImage>> {
    let tiles_path = Path::new("tiles/");
    let mut tiles = Vec::new();

    for entry_result in tiles_path.read_dir()? {
        let entry = entry_result?;
        let image = open(entry.path())?;
        tiles.push(image.resize(50,50,FilterType::Nearest));
    }

    Ok(tiles)
}

// Partition and calculate the average color of each image
fn partition(tiles: Vec<DynamicImage>) -> (Vec<AverageColor>,Vec<AverageColor>) {
    let (left, mut right): (Vec<AverageColor>, Vec<AverageColor>) = tiles.into_par_iter()
        .partition_map(|tile|  {
            let (average_color,homogenous) = {
                let (color,histo) = get_average_color(&tile);
                (color,is_homogenous(histo))
            };
          
            if homogenous { Either::Left(AverageColor::Homogenous{ image: tile,color: average_color }) }
            else { Either::Right(AverageColor::Non{image: tile,color: average_color, edges: None })}
        });

    for im in right.iter_mut() {
        if let AverageColor::Non { ref image, ref mut edges,..} = *im {
            *edges = Some(edge_map(image));
        }
    }
    (left,right)
}

fn get_average_color(image: &DynamicImage) -> ([u8;3], RGBHistogram) {
    let mut rgb: [usize;3] = [0;3];
    let mut r_histogram: [u32;256] = [0;256];
    let mut g_histogram: [u32;256] = [0;256];
    let mut b_histogram: [u32;256] = [0;256];
    for p in image.pixels() {
        let pix = p.2.to_rgb();
        let channels = pix.channels();

        let red = channels[0] as usize;
        let green = channels[1] as usize;
        let blue = channels[2] as usize;

        rgb[0] += red;
        r_histogram[red] += 1;

        rgb[1] += green;
        g_histogram[green] += 1;
        
        rgb[2] += blue;
        b_histogram[blue] += 1;
        }
    let p_count = (image.width() * image.height()) as usize;
    for i in 0..rgb.len() {
        rgb[i] = rgb[i] / p_count;
    }

    ([rgb[0] as u8, rgb[1] as u8,rgb[2] as u8], 
        RGBHistogram{r_histogram,g_histogram,b_histogram,image})
}

// Select the tile that is the closest match to out target RGB color value
fn nearest<'t>(target :AverageColor, tiles: &'t Vec<AverageColor>) -> &'t DynamicImage {
    let mut nearest_tile = tiles[0].get_image();
    let mut smallest_dist = f64::MAX;

    for tile in tiles {
        if let AverageColor::Non { ref edges,..} = target {
            let target_edges = edges.as_ref().unwrap();
            if let AverageColor:: Non { ref edges,.. } = *tile {
                let tile_edges = edges.as_ref().unwrap();
                if !edge_map_compare(target_edges,tile_edges) { continue }
            }
        }
        let dist = distance(target.get_color(),tile.get_color());
        if dist < smallest_dist { 
            smallest_dist = dist;
            nearest_tile = tile.get_image();
        }
    }

    nearest_tile
}

// Euclidean distance between 2 RGB color values
fn distance(p1: &[u8], p2: &[u8]) -> f64 {
    let square = |x| x * x;
    let euclidean_squared = (square(p1[0] as i32 - p2[0] as i32) 
        + square(p1[1] as i32 - p2[1] as i32) 
        + square(p1[2] as i32 - p2[2] as i32)) as f64;

    euclidean_squared
}

// Analyze a RGBHistogram to determine if an image is made up of mostly one color
fn is_homogenous(histo: RGBHistogram) -> bool {
    let image = histo.image;

    let max_index = |histo: [u32;256]| -> usize {
        let mut i = 0;
        for (j, &val) in histo.iter().enumerate() {
            if val > histo[i] {
                i = j;
            }
        }
        i
    };

    let red_max = max_index(histo.r_histogram);
    let green_max = max_index(histo.g_histogram);
    let blue_max = max_index(histo.b_histogram);

    let range = |percent,val| -> (u32,u32) {
        let range_percent = (256 * percent / 100) as u32;

        let lo = utils::clamp(val - range_percent, 0,255);
        let hi = utils::clamp(val + range_percent, 0,255);

        (lo,hi)
    };

    let red_range = range(30,red_max as u32);
    let green_range = range(30,green_max as u32);
    let blue_range = range(30,blue_max as u32);

    let pixel_count = (image.width() * image.height()) * 80 / 100;
    let (mut pixels_in_red, mut pixels_in_green,mut pixels_in_blue) = (0,0,0);

    for p in image.pixels() {
        let pix = p.2.to_rgb();
        let channels = pix.channels();

        let (red,green,blue) = (channels[0] as u32,channels[1] as u32,channels[2] as u32);

        if red >= red_range.0 && red <= red_range.1 {
            pixels_in_red += 1;
        }

        if green >= green_range.0 && green <= green_range.1 {
            pixels_in_green += 1;
        }

        if blue >= blue_range.0 && blue <= blue_range.1 {
            pixels_in_blue += 1;
        }
    }

    pixels_in_red >= pixel_count && pixels_in_green >= pixel_count && pixels_in_blue >= pixel_count
}

/*
* Compares the number of different pixels in 2 grayscale images.
* Returns true if the number of different pixels is less than or equal to a 15%
* of the total pixels.
*/
fn edge_map_compare(left: &GrayImage, right: & GrayImage) -> bool {
    //assert_eq!(left.dimensions(),right.dimensions(), "Error comparing edge maps: unequal dimensions");

    let max_pixel_diff = (left.width() * left.height() * 15) / 100; 
    let mut total_diff = 0;
    for (a, b) in left.pixels().zip(right.pixels()) {
        if a.channels() != b.channels() { total_diff += 1; };
    }

    total_diff <= max_pixel_diff
}

fn edge_map(image: &DynamicImage) -> GrayImage {
    let kernel = [
        1.0/9.0,1.0/9.0,1.0/9.0,
        1.0/9.0,1.0/9.0,1.0/9.0,
        1.0/9.0,1.0/9.0,1.0/9.0
    ];
    let gray_image = colorops::grayscale(&image.filter3x3(&kernel));

    let mut average = {
        let mut sum = 0;
        for p in gray_image.pixels() {
            sum += p.channels()[0] as u32;
        }

        (sum / (gray_image.width() * gray_image.height())) as f32
    };

    if average == 0.0 { average = 255.0 }
    let lo = average*0.66;
    let hi = average*1.33;
    
    edges::canny(&gray_image,lo,hi)
}

#[cfg(test)]
mod test {
    use super::distance;

    #[test]
    fn test_distance() {
        let square = |x| x*x;

        let black = &[0,0,0];
        let white = &[255,255,255];

        let p = &[121,30,177];
        let q = &[237,22,88];

        assert!(f64::abs(distance(black,white).floor() - square(441.672956)) <= 1.0);
        assert!(f64::abs(distance(p,q).floor() - square(146.427456)) <= 1.0);
    }

    #[test]
    fn test_distance_between_same_point() {
        let p1 = &[2,3,5];
        let p2 = &[235,110,75];

        assert_eq!(distance(p1,p1),0.0);
        assert_eq!(distance(p2,p2),0.0);
    }
}
