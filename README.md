# Photomosaic

A photomosaic generator written in Rust

[DEMO](https://secure-ocean-69604.herokuapp.com/)

![Example](https://dl.dropboxusercontent.com/s/jimrm0iksvbh52o/city-hall-philly_comparison.png?dl=0)

### How it works

Scans the directory provided as input for images to resize and calculate their mean color values. Then, it partitions the images; those with distinct edges are separated from images that are made of mostly 1 color. The input image is split into cells  and a tile selected to replace a cell is selected by minimizing the distance between their mean color values and/or their edge maps.

### Command-line Usage

#### (Build from source)
First, [install Rust](https://www.rust-lang.org/en-US/install.html).

Next, clone this repository and navigate to the new directory

run

    cargo build --release
    .\target\release\photomosaic.exe --help

| Flag | Default | Description |
| --- | --- | --- |
| `i` | n/a | input file |
| `o` | n/a | output file |
| `s` | 30 | size of images that are used as tiles in the mosaic. Tile dimensions are s x s |
| `t` | n/a | Path to a directory containing images that are used as tiles |


### Tips

Provide a large pool of images (1000+)

Use a smaller size for tiles if you want a mosaic with more detail. Experiment with different sizes and images

### Examples

![Example](https://dl.dropboxusercontent.com/s/ius0l1q7twcvvuh/mosaic_comparison_goku.png?dl=0)
![Example](https://dl.dropboxusercontent.com/s/jd7ru2r3rhm6hhj/owl_comparison.png?dl=0)
