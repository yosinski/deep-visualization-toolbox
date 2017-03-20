#!/usr/bin/env bash

# Exit on first error
set -e

function fetch_file() {
    filename="$1"
    url="$2"
    if [ -e "$filename" ]; then
        echo "$url: file already downloaded (remove $filename to force re-download)"
    else
        echo "$url: fetching..."
        wget -O "$filename" "$url"
        echo "$url: done."
    fi
}

function fetch_and_extract() {
    filename="$1"
    url="$2"
    dir="$3"
    example_filename="$4"
    example_path="$dir/$example_filename"
    if [ -e "$example_path" ]; then
        echo "$url: $example_path already exists, skipping."
    else
        fetch_file "$filename" "$url"
        echo "$url: extracting..."
        mkdir -p "$dir"
        tar -C "$dir" -xzf "$filename"
        echo "$url: done."
    fi
}

fetch_file caffenet-yos-weights http://c.yosinski.com/caffenet-yos-weights
fetch_file ilsvrc_2012_mean.npy http://c.yosinski.com/ilsvrc_2012_mean.npy

fetch_and_extract caffenet-yos_regularized_opt.tar.gz http://c.yosinski.com/caffenet-yos_regularized_opt.tar.gz   unit_jpg_vis regularized_opt/conv1/conv1_0000_montage.jpg
fetch_and_extract caffenet-yos_max_im.tar.gz          http://c.yosinski.com/caffenet-yos_max_im.tar.gz             unit_jpg_vis max_im/conv1/conv1_0000.jpg
fetch_and_extract caffenet-yos_max_deconv.tar.gz      http://c.yosinski.com/caffenet-yos_max_deconv.tar.gz         unit_jpg_vis max_deconv/conv1/conv1_0000.jpg

if [ "$1" = "all" ]; then
    fetch_and_extract caffenet-yos_regularized_opt_fc6_fc7.tar.gz http://c.yosinski.com/caffenet-yos_regularized_opt_fc6_fc7.tar.gz unit_jpg_vis regularized_opt/fc6/fc6_0000_montage.jpg
    fetch_and_extract caffenet-yos_max_im_fc6_fc7.tar.gz          http://c.yosinski.com/caffenet-yos_max_im_fc6_fc7.tar.gz          unit_jpg_vis max_im/fc6/fc6_0000.jpg
    fetch_and_extract caffenet-yos_max_deconv_fc6_fc7.tar.gz      http://c.yosinski.com/caffenet-yos_max_deconv_fc6_fc7.tar.gz      unit_jpg_vis max_deconv/fc6/fc6_0000.jpg
else
    echo
    echo "Rerun as \"$0 all\" to also fetch fc6 and fc7 unit visualizations (Warning: 4.5G more)"
fi
