#!/usr/bin/env python3
"""Sort EPU AFIS data into Optics Groups for RELION 5.0

Written By Dustin Reed Morado
Last updated 20.06.2020
"""

import os
import argparse

from collections.abc import Iterable, Sequence
from fnmatch import fnmatch
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree

STAR_VERSION_OLD = 30001
STAR_VERSION = 50001


def _parse_metadata_file(metadata_path: str | os.PathLike[str],
                         namespace: dict[str, str]) -> list[float]:
    metadata_path = os.fspath(metadata_path)
    metadata = ElementTree.parse(metadata_path)
    beam_shift = metadata.findall(
        'fei:microscopeData/fei:optics/fei:BeamShift', namespace)

    if len(beam_shift) == 0:
        raise ValueError(f'No BeamShift found in {metadata_path}')
    elif len(beam_shift[0]) != 2:
        raise ValueError(f'Improper BeamShift found in {metadata_path}')

    return [float(x.text) for x in beam_shift[0]]


def _index_movie_files(movie_dir: str, ftype: str,
                       metadata_roots: Iterable[str]) -> dict[str, str]:
    movie_dir = os.path.abspath(movie_dir)
    indexed: dict[str, str] = {}
    roots = [root for root in metadata_roots if root]

    for dirpath, _, filenames in os.walk(movie_dir, followlinks=True):
        for filename in filenames:
            if not filename.endswith(f'.{ftype}'):
                continue

            lower_name = filename.lower()
            for root in roots:
                if root and root in filename and 'ractions' in lower_name:
                    if root not in indexed:
                        indexed[root] = os.path.join(dirpath, filename)
                    break

    return indexed


def _configure_sklearn_threads(thread_count: int) -> None:
    thread_count = int(thread_count)
    if thread_count < 1:
        raise ValueError('thread_count must be positive')

    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = str(thread_count)


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
            description = "Create Optics Groups from EPU AFIS data",
            epilog = "Writted by Dustin Morado 20.06.2020",
            suggest_on_error = True)

    parser.add_argument('--xml_dir', '-i', type = str, required = True,
            help = ('Path to directory with EPU record metadata XML files. '
                    '[REQUIRED]'))

    parser.add_argument('--n_clusters', '-n', type = int, default = 1,
            help = 'Number of Optics Groups (1 = interactively choose. [1]')

    parser.add_argument('--algorithm', type = str, default = 'kmeans',
            choices = ['kmeans', 'hac'],
            help = 'Clustering algorthim to use. {kmeans, hac} [kmeans]')

    parser.add_argument('--apix', type = float, default = 1.00,
            help = 'Pixel size of micrographs. [1.00]')

    parser.add_argument('--mtf_fn', type = str, default = 'MTF.star',
            help = 'Path to MTF STAR file. [MTF.star]')

    parser.add_argument('--voltage', type = float, default = 300.0,
            help = 'Voltage of micrscope in keV. [300.0]')

    parser.add_argument('--cs', type = float, default = 2.7,
            help = 'Spherical Aberration of objective lens in mm. [2.7]')

    parser.add_argument('--q0', type = float, default = 0.1,
            help = 'Fraction of amplitude contrast [0 - 1]. [0.1]')

    parser.add_argument('--ftype', type = str, default = 'mrc',
            help = 'Filetype of micrograph movies. [mrc]')

    parser.add_argument('--movie_dir', '-d', type = str, default = '.',
            help = 'Path to directory with micrograph movies. [.]')

    parser.add_argument('--movie_suffix', type = str, default = 'Fractions',
            help = 'Suffix to append to micrograph movie filenames. [Fractions]')

    parser.add_argument('--output_fn', '-o', type = str,
            default = 'movies.star',
            help = 'Path to output STAR file. [movies.star]')

    parser.add_argument('--quiet', '-q', action = 'store_true',
            help = 'Run non-interactively with no plotting. [False]')

    parser.add_argument('--workers', type = int, default = 1,
            help = 'Number of threads used for XML parsing and for scikit-learn internals. [1]')

    args = parser.parse_args(argv)
    main(xml_dir = args.xml_dir,
         n_clusters = args.n_clusters,
         apix = args.apix,
         mtf_fn = args.mtf_fn,
         voltage = args.voltage,
         cs = args.cs,
         q0 = args.q0,
         ftype = args.ftype,
         movie_dir = args.movie_dir,
         movie_suffix = args.movie_suffix,
         output_fn = args.output_fn,
         algorithm = args.algorithm,
         quiet = args.quiet,
         workers = args.workers)
    return 0


def main(xml_dir: str | None = None, n_clusters: int = 1, apix: float = 1.00,
         mtf_fn: str = 'MTF.star', voltage: float = 300.0, cs: float = 2.7,
         q0: float = 0.1, ftype: str = 'mrc', movie_dir: str = '',
         movie_suffix: str = 'Fractions', output_fn: str = 'movies.star',
         algorithm: str = 'kmeans', quiet: bool = False,
         workers: int | None = 1) -> None:

    if xml_dir is None:
        xml_dir = os.getcwd()

    metadata_fns: list[str] = []

    for dirpath, _, filenames in os.walk(xml_dir, followlinks=True):
        for filename in filenames:
            if (fnmatch(filename, 'FoilHole_*_Data_*.xml')
                    and not fnmatch(filename, '*ractions.xml')):

                metadata_fns.append(os.path.join(dirpath, filename))

    if not metadata_fns:
        raise ValueError('No EPU XML metadata files found')

    namespace = {'fei' :
            'http://schemas.datacontract.org/2004/07/Fei.SharedObjects'}

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans, AgglomerativeClustering

    if workers is not None:
        _configure_sklearn_threads(workers)

    metadata_roots = [os.path.splitext(os.path.basename(fn))[0]
                      for fn in metadata_fns]
    movie_lookup = _index_movie_files(movie_dir, ftype, metadata_roots)

    beam_shifts = []
    if workers and int(workers) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            parser = partial(_parse_metadata_file, namespace=namespace)
            beam_shifts = list(executor.map(parser, metadata_fns))
    else:
        beam_shifts = [_parse_metadata_file(metadata_fn, namespace)
                      for metadata_fn in metadata_fns]

    shift_array = np.array(beam_shifts)

    if n_clusters == 1:
        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.set_title('EPU AFIS Beam Shift Values')
        axes.set_xlabel('Beam Shift X (a.u.)')
        axes.set_ylabel('Beam Shift Y (a.u.)')
        axes.scatter(shift_array[:, 0], shift_array[:, 1])
        plt.show(block = False)

        while n_clusters <= 1:
            n_user = input('How many Optics Groups (or Ctrl-C to abort): ')

            try:
                n_clusters = int(n_user)
            except ValueError:
                print('Please enter a positive integer greater than 1!')
            else:
                if n_clusters <=1:
                    print('Please enter a positive integer greater than 1!')

        plt.close('all')

    if len(beam_shifts) < n_clusters:
        raise ValueError('Number of clusters greater than number of points')

    optics_clusters = None
    cluster_centers = None

    if algorithm == 'kmeans':
        k_means = KMeans(n_clusters = n_clusters,
                         init = 'k-means++',
                         n_init = 100,
                         algorithm = 'elkan').fit(shift_array)

        optics_clusters = k_means.predict(shift_array)
        cluster_centers = k_means.cluster_centers_
    else:
        hac = AgglomerativeClustering(n_clusters = n_clusters,
                                      linkage = 'complete').fit(shift_array)

        optics_clusters = hac.labels_
        cluster_centers = [[np.average(shift_array[optics_clusters==x][:,0]),
                            np.average(shift_array[optics_clusters==x][:,1])]
                           for x in np.unique(optics_clusters)]

    radii = [x[0]**2 + x[1]**2 for x in cluster_centers]
    sort_idxs = sorted(enumerate(radii), key=lambda x:x[1])
    origin = cluster_centers[sort_idxs[0][0]]

    x_vec = (1, 0)
    x_unit = (1, 0)
    x_angle = np.pi
    x_length = 1

    y_vec = (0, 1)
    y_unit = (0, 1)
    y_angle = np.pi
    y_length = 1

    for idx in range(1, min(5, len(cluster_centers))):
        point = cluster_centers[sort_idxs[idx][0]]
        x_coord = point[0] - origin[0]
        y_coord = point[1] - origin[1]

        if x_coord > 0 and x_coord > abs(y_coord):
            x_vec = (x_coord, y_coord)
            x_angle = np.arctan2(y_coord, x_coord)
            x_length = np.sqrt(x_vec[0]**2 + x_vec[1]**2)
            x_unit = (x_vec[0] / x_length, x_vec[1] / x_length)


        if y_coord > 0 and y_coord > abs(x_coord):
            y_vec = (x_coord, y_coord)
            y_angle = np.arctan2(y_coord, x_coord)
            y_length = np.sqrt(y_vec[0]**2 + y_vec[1]**2)
            y_unit = (y_vec[0] / y_length, y_vec[1] / y_length)

    grid_dists = []

    for idx, center in enumerate(cluster_centers):
        point = (center[0] - origin[0], center[1] - origin[1])
        grid_x = np.round(
            (point[0] * x_unit[0] + point[1] * x_unit[1]) / x_length)

        grid_y = np.round(
            (point[0] * y_unit[0] + point[1] * y_unit[1]) / y_length)

        dist = max(abs(grid_x), abs(grid_y))
        angle = (np.degrees(np.arctan2(grid_y, grid_x)) + 360) % 360
        grid_dists.append((idx, grid_x, grid_y, dist, angle))

    sort_idxs = [x[0] for x in sorted(grid_dists, key=lambda x:(x[3], x[4]))]
    optics_groups = [sort_idxs.index(x) + 1 for x in optics_clusters]

    if not quiet:
        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.set_title('EPU AFIS Beam Shift Values')
        axes.set_xlabel('Beam Shift X (a.u.)')
        axes.set_ylabel('Beam Shift Y (a.u.)')
        axes.scatter(shift_array[:, 0], shift_array[:, 1], c=optics_groups,
                cmap='tab20')

        for optics_group in range(n_clusters):
            axes.annotate(f'{optics_group + 1:d}',
                    xy = cluster_centers[sort_idxs[optics_group]],
                    textcoords = 'offset pixels', xytext=(10, 10))

        plt.show(block = False)
        _ = input('If happy with cluster press any key (Ctrl-C to abort)')
        plt.close('all')

    entries = sorted(zip(metadata_fns, optics_groups, strict=True),
            key=lambda x:(x[0], x[1]))

    with open(output_fn, 'w', encoding='utf-8') as f:
        f.write(f'\n# version {STAR_VERSION:d}\n\n')
        f.write('data_optics\n\nloop_\n')
        f.write('_rlnOpticsGroupName #1 \n')
        f.write('_rlnOpticsGroup #2 \n')
        f.write('_rlnMtfFileName #3 \n')
        f.write('_rlnMicrographOriginalPixelSize #4 \n')
        f.write('_rlnVoltage #5 \n')
        f.write('_rlnSphericalAberration #6 \n')
        f.write('_rlnAmplitudeContrast #7 \n')

        for optics_group in range(1, n_clusters + 1):
            f.write(f'opticsGroup{optics_group:d}\t{optics_group:d}\t'
                    f'{mtf_fn}\t{apix:15.6f}\t{voltage:15.6f}\t'
                    f'{cs:15.6f}\t{q0:15.6f}\n')

        f.write(f' \n\n# version {STAR_VERSION:d}\n\n')
        f.write('data_movies\n\nloop_\n')
        f.write('_rlnMicrographMovieName #1 \n')
        f.write('_rlnOpticsGroup #2 \n')

        for metadata_fn, optics_group in entries:
            base = os.path.basename(metadata_fn)
            root, _ = os.path.splitext(base)
            movie_fn = None

            if root in movie_lookup:
                movie_fn = movie_lookup[root]
            else:
                movie_glob = f'*{root}*ractions.{ftype}'

                for dirpath, _, filenames in os.walk(movie_dir, followlinks=True):
                    for filename in filenames:
                        if fnmatch(filename, movie_glob):
                            movie_fn = os.path.join(dirpath, filename)
                            break

                    if movie_fn is not None:
                        break

            if movie_fn is None:
                movie_fn = (f'{os.path.join(movie_dir, root)}_'
                            f'{movie_suffix}.{ftype}')

            f.write(f'{movie_fn}\t{optics_group:d}\n')

        f.write(' \n')


if __name__ == '__main__':
    raise SystemExit(cli())
