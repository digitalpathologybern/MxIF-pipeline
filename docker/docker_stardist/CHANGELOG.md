# Changelog of image "stardist"

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

### Fixed

## [v1.0.2] - 2024-12-04
### Added
- Most recent release of the stardist jupyter image.
- Base Dockerfile taken from [the stardist github](https://github.com/stardist/stardist/tree/main/docker) using `tensorflow/tensorflow:2.11.0-gpu-jupyter`.

## [MxIF-pipeline-v3] - 24-12-05
### Added
- Packages needed for the "MxIF-pipeline" workflow are added using "requirements.txt":
    - numpy==1.24.4
    - csbdeep==0.8.1
    - shapely==2.0.6
    - geopandas==0.13.2
    - pandas==2.0.3
    - opencv-python-headless==4.10.0.84
    - tifffile==2023.7.10
    - rasterio==1.3.11
    - xtiff==0.7.9
    - scikit-image==0.21.0
    - scipy==1.10.1
    - matplotlib==3.7.5
    - anndata==0.9.2

