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

fetch_file bvlc-googlenet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
