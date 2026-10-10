# EPU_GROUP_AFIS

Takes the XML metadata from an EPU session that used AFIS (_Aberration Free
Image Shift_) for faster acquisition and organizes the collected micrograph
movies into RELION Optics Groups. In RELION version 3.1 where these groups were
introduced they can be used to further reduce the effects of high-order
aberrations on the data induced by collecting using large amounts of beam-image
shift. The output `movies.star` uses the RELION 5.0 STAR format
(`# version 50001`), matching what `relion_import` writes.

There is a Notebook here that goes through how the processing is done and then a
script that is for actual processing.

## Requirements

The script targets Python 3.14 and newer:

 * Python (_Version >= 3.14_)
 * numpy
 * scikit-learn
 * matplotlib

## Installation

Install the package in editable mode from the repository root so the command becomes available as `epu_group_afis`:

```bash
python -m pip install -e .
```

If you prefer a regular installation instead of editable mode, use:

```bash
python -m pip install .
```

After installation, you can invoke the tool directly as:

```bash
epu_group_afis --xml_dir /path/to/xml --quiet
```

## Usage
```
usage: epu_group_afis [-h] --xml_dir XML_DIR [--n_clusters N_CLUSTERS]
                      [--algorithm {kmeans,hac}] [--apix APIX]
                      [--mtf_fn MTF_FN] [--voltage VOLTAGE] [--cs CS]
                      [--q0 Q0] [--ftype FTYPE] [--movie_dir MOVIE_DIR]
                      [--movie_suffix MOVIE_SUFFIX] [--output_fn OUTPUT_FN]
                      [--quiet] [--workers WORKERS]

Create Optics Groups from EPU AFIS data

options:
  -h, --help            show this help message and exit
  --xml_dir, -i XML_DIR
                        Path to directory with EPU record metadata XML files.
                        [REQUIRED]
  --n_clusters, -n N_CLUSTERS
                        Number of Optics Groups (1 = interactively choose. [1]
  --algorithm {kmeans,hac}
                        Clustering algorthim to use. {kmeans, hac} [kmeans]
  --apix APIX           Pixel size of micrographs. [1.00]
  --mtf_fn MTF_FN       Path to MTF STAR file. [MTF.star]
  --voltage VOLTAGE     Voltage of micrscope in keV. [300.0]
  --cs CS               Spherical Aberration of objective lens in mm. [2.7]
  --q0 Q0               Fraction of amplitude contrast [0 - 1]. [0.1]
  --ftype FTYPE         Filetype of micrograph movies. [mrc]
  --movie_dir, -d MOVIE_DIR
                        Path to directory with micrograph movies. [.]
  --movie_suffix MOVIE_SUFFIX
                        Suffix to append to micrograph movie filenames.
                        [Fractions]
  --output_fn, -o OUTPUT_FN
                        Path to output STAR file. [movies.star]
  --quiet, -q           Run non-interactively with no plotting. [False]
  --workers WORKERS     Number of threads used for XML parsing and for scikit-
                        learn internals. [1]

Writted by Dustin Morado 20.06.2020
```

## Licensing

Distributed under the MIT License please see LICENSE.txt for more information.
