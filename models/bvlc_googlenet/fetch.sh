#!/usr/bin/env bash

# Exit on first error
set -e

function fetch_file() {
    url_base="$1"
    filename="$2"
    url="$url_base/$filename"
    if [ -e "$filename" ]; then
        echo "$url: file already downloaded (remove $filename to force re-download)"
    else
        echo "$url: fetching..."
        wget "$url"
        echo "$url: done."
    fi
}

function fetch_and_extract() {
    url_base="$1"
    filename="$2"
    dir="$3"
    example_filename="$4"
    url="$url_base/$filename"
    example_path="$dir/$example_filename"
    if [ -e "$example_path" ]; then
        echo "$url: $example_path already exists, skipping."
    else
        fetch_file "$url_base" "$filename"
        echo "$url: extracting..."
        mkdir -p "$dir"
        tar -C "$dir" -xzf "$filename"
        echo "$url: done."
    fi
}

fetch_file http://dl.caffe.berkeleyvision.org/ bvlc_googlenet.caffemodel
