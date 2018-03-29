#[macro_use]
extern crate structopt;
extern crate image;
extern crate imageproc;
extern crate rayon;

use std::path::{Path, PathBuf};
use std::f64;
use image::{DynamicImage, open, FilterType, GenericImage, Pixel,RgbaImage,GrayImage};
use image::math::utils;
use image::imageops::colorops;
use imageproc::edges;
use rayon::prelude::*;
use rayon::iter::Either;
use structopt::StructOpt;

type GenError = Box<std::error::Error>;
type GenResult<T> = Result<T, GenError>;

#[derive(StructOpt, Debug)]
#[structopt(name = "photomosaic")]
struct Opt {
    /// Input file
    #[structopt(short = "i", long = "input", parse(from_os_str))]
    input: PathBuf,

    /// Output file, image format is based on the extension
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: PathBuf,

    /// Directory containing images to be used as tiles
    #[structopt(short = "t", long = "tiles", parse(from_os_str))]
    tiles_dir: PathBuf,

    /// Size of tile images
    #[structopt(short = "s", long = "size", default_value = "30")]
    tile_size: u32
}

enum AverageColor {
    Homogenous { image: DynamicImage, color: [u8;3] },
    Non { image: DynamicImage, color: [u8;3], edges: Option<GrayImage> }
}

impl AverageColor {
    fn get_image(&self) -> &DynamicImage {
        match *self {
            AverageColor::Homogenous { ref image,.. } | AverageColor::Non { ref image,.. } => image,
        }
    }

    fn get_color(&self) -> &[u8] {
        match *self {
            AverageColor::Homogenous { ref color,.. } | AverageColor::Non { ref color,.. } => color,
        }
    }
}

struct RGBHistogram<'im> {
    r_histogram: [u32;256],
    g_histogram: [u32;256],
    b_histogram: [u32;256],
    image: &'im DynamicImage
}

fn main() {
    let opt = Opt::from_args();

    if opt.output.exists() {
        eprintln!("Error: an entry already exists at the output path");
        ::std::process::exit(1);
    }

    let original = open(opt.input)
        .expect("Error opening target image");
    
    let raw_tiles = load_tiles(&opt.tiles_dir,opt.tile_size).expect("Error opening tiles directory");
    let resized_tiles = partition(raw_tiles);
    create_mosaic(original,opt.output,resized_tiles,opt.tile_size);
}

// Create photomosaic a based on an original image formed from tiles
fn create_mosaic<P>(mut original: DynamicImage,output: P,tiles: (Vec<AverageColor>,Vec<AverageColor>), tile_size: u32)
    where P: AsRef<Path>
{
    let (width, height) = original.dimensions();
    let (homo, edges) = tiles;
    
    let mut new_image = RgbaImage::new(width,height);

    let (mut x,mut y) = (0,0);
    while y < height {
        x = 0;
        while x < width {
            let cell = original.crop(x,y,tile_size,tile_size);
            let (average_color,homogenous) = {
                let (color,histo) = get_average_color(&cell);
                (color,is_homogenous(histo))
            };

            let nearest_tile = if homogenous {
                nearest(&AverageColor::Homogenous { image: cell, color: average_color } ,&homo)
            } else {
                nearest(&AverageColor::Non { edges: Some(edge_map(&cell)), image: cell, color: average_color }
                    ,&edges)
            };

            new_image.copy_from(nearest_tile,x,y);
            x += tile_size;
        }
        y += tile_size
}

    new_image.save(output).expect("Error saving new_mosaic.png");
}

// Load and resize images to be used as tiles in a mosaic
fn load_tiles(tiles_path: &PathBuf, tile_size: u32) -> GenResult<Vec<DynamicImage>> {
    let mut tiles = Vec::new();
    
    for entry_result in tiles_path.read_dir()? {
        let entry = entry_result?;
        let image = open(entry.path())?;
        tiles.push(image.resize_exact(tile_size,tile_size,FilterType::Nearest));
    }

    Ok(tiles)
}

// Partition and calculate the average color of each image
fn partition(tiles: Vec<DynamicImage>) -> (Vec<AverageColor>,Vec<AverageColor>) {
    let (homo, mut edges): (Vec<AverageColor>, Vec<AverageColor>) = tiles.into_par_iter()
        .partition_map(|tile|  {
            let (average_color,homogenous) = {
                let (color,histo) = get_average_color(&tile);
                (color,is_homogenous(histo))
            };
          
            if homogenous { Either::Left(AverageColor::Homogenous{ image: tile,color: average_color }) }
            else { Either::Right(AverageColor::Non{image: tile,color: average_color, edges: None })}
        });

    for im in &mut edges {
        if let AverageColor::Non { ref image, ref mut edges,..} = *im {
            *edges = Some(edge_map(image));
        }
    }
    (homo,edges)
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
    for chan in &mut rgb {
        *chan /= p_count;
    }

    ([rgb[0] as u8, rgb[1] as u8,rgb[2] as u8], 
        RGBHistogram{r_histogram,g_histogram,b_histogram,image})
}

// Select the tile that is the closest match to out target RGB color value
fn nearest<'t>(target: &AverageColor, tiles: &'t [AverageColor]) -> &'t DynamicImage {
    let mut nearest_tile = tiles[0].get_image();
    let mut smallest_dist = f64::MAX;

    for tile in tiles {
        if let AverageColor::Non { ref edges,..} = *target {
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
    f64::from(square(i32::from(p1[0]) - i32::from(p2[0])) 
        + square(i32::from(p1[1]) - i32::from(p2[1])) 
        + square(i32::from(p1[2]) - i32::from(p2[2])))

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

        let (red,green,blue) = (u32::from(channels[0]),u32::from(channels[1]),u32::from(channels[2]));

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
            sum += u32::from(p.channels()[0]);
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
