"""
Module that wraps some legacy code to get a set of snapshots and domain
decompose given slug flow problem.

Code is not very general and likely only works for exact slug flow data
used in this project. Note this code is meant to be a wrapper for legacy code
that is intended to not be used used very often or in a critical/production
setting. Therefore sustainability  may be lacking.
"""

import numpy as np
import sys
import argparse

if sys.version_info[0] < 3:
    import u2r # noqa
else:
    import u2rpy3 # noqa
    u2r = u2rpy3

sys.path.append("/usr/lib/python2.7/dist-packages/")
import vtktools # noqa

__author__ = "Claire Heaney, Zef Wolffs"
__credits__ = ["Jon Atli Tomasson"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Zef Wolffs"
__email__ = "zefwolffs@gmail.com"
__status__ = "Development"


def get_snapshots_3D(
    random=True,
    nfiles=2,
    offset=0,
    ndatapoints=20,
    in_file_base="slug_255_exp_projected_",
    out_file="sf_snapshots.npy",
):
    """
    Get snapshots from slug flow 3D dataset. Note that this function also
    randomly selects along the axial axis `ndatapoints` number of subdomains
    per vtu file. Stores results in out_file numpy file.

    Args:
        random (bool): If true, select random samples, otherwise split grid
                       10-fold
        nfiles (int): Number of vtu files (starting from 0)
        ndatapoints (int): Number of random subdomains to sample per vtu file
        out_file (string): Output numpy filename
    """

    # hardwire for lazyness
    # only checking one time level and x component of velocity here
    nscalar_velocity = 3
    nscalar_alpha = 1
    ndim = 3
    nTime = 1
    nloc = 4

    # info from vtu file - has DG velocities
    for k in range(nfiles):
        print("k: ", k)
        filename = in_file_base + str(k) + ".vtu"
        vtu_data = vtktools.vtu(filename)
        velocity = vtu_data.GetField("phase1::Velocity")
        alpha = vtu_data.GetField("Component1::ComponentMassFractionPhase1")
        print("shape velocity", velocity.shape)
        print("shape alpha", alpha.shape)
        coordinates = vtu_data.GetLocations()

        nNodes = coordinates.shape[0]  # vtu_data.ugrid.GetNumberOfPoints()
        print("nNodes", nNodes)
        nEl = vtu_data.ugrid.GetNumberOfCells()
        print("nEl", nEl, type(nEl))  # 6850

        x_all = np.transpose(coordinates[:, 0:3])

        if not random:
            ndatapoints = 10

        for i in range(offset, offset+ndatapoints):
            velocity_mesh = np.zeros(
                (nscalar_velocity, nNodes, nTime)
            )  # value_mesh(nscalar,nonods,ntime)
            velocity_mesh[:, :, 0] = np.swapaxes(velocity[:, :], 0, 1)

            alpha_mesh = np.zeros(
                (nscalar_alpha, nNodes, nTime)
            )  # value_mesh(nscalar,nonods,ntime)
            alpha_mesh[0, :, 0] = alpha[:, 0]

            # rectangular domain so
            y0 = min(coordinates[:, 1])
            z0 = min(coordinates[:, 2])
            yN = max(coordinates[:, 1])
            zN = max(coordinates[:, 2])

            # Let's pick a random block along x axis of 1m length
            if random:
                x0 = float(np.random.randint(0, 9000)) / 1000
                xN = x0 + 1
            else:
                x0 = float(i)
                xN = x0 + 1

            # print('(x0,y0,z0)',x0, y0, z0)
            # print('(xN,yN,zN)',xN, yN, zN)
            block_x_start = np.array((x0, y0, z0))
            # block_x_start = np.array(( 0, 0, 10 ))

            # get global node numbers
            x_ndgln = np.zeros((nEl * nloc), dtype=int)
            for iEl in range(nEl):
                n = vtu_data.GetCellPoints(iEl) + 1
                x_ndgln[iEl * nloc: (iEl + 1) * nloc] = n

            # We set these values hard, TODO: change to input variables
            # set grid size
            nx = 60  # 512#128
            ny = 20  # 512#128
            nz = 20  # 128#32

            ylength = 0.078
            zlength = 0.078
            xlength = 1  # length of a subdomain
            ddx = np.array([float(xlength) / (nx - 1), ylength / (ny - 1),
                           zlength / (nz - 1)])

            # print('nx, ny, nz', nx, ny, nz)
            # print('dx, dy, dz', ddx)
            ###################################################################
            # print('np.max(velocity_mesh)', np.max(velocity_mesh))
            # print('np.min(velocity_mesh)', np.min(velocity_mesh))

            # interpolate from (unstructured) mesh to (structured) grid
            zeros_outside_mesh = 0
            velocity_grid = u2r.simple_interpolate_from_mesh_to_grid(
                velocity_mesh,
                x_all,
                x_ndgln,
                ddx,
                block_x_start,
                nx,
                ny,
                nz,
                zeros_outside_mesh,
                nEl,
                nloc,
                nNodes,
                nscalar_velocity,
                ndim,
                nTime,
            )

            # print('np.max(velocity_grid)', np.max(velocity_grid))
            # print('np.min(velocity_grid)', np.min(velocity_grid))

            ###################################################################
            # print('np.max(alpha_mesh)', np.max(alpha_mesh))
            # print('np.min(alpha_mesh)', np.min(alpha_mesh))

            # interpolate from (unstructured) mesh to (structured) grid
            zeros_outside_mesh = 0
            alpha_grid = u2r.simple_interpolate_from_mesh_to_grid(
                alpha_mesh,
                x_all,
                x_ndgln,
                ddx,
                block_x_start,
                nx,
                ny,
                nz,
                zeros_outside_mesh,
                nEl,
                nloc,
                nNodes,
                nscalar_alpha,
                ndim,
                nTime,
            )

            # print('np.max(alpha_grid)', np.max(alpha_grid))
            # print('np.min(alpha_grid)', np.min(alpha_grid))

            velocity_array = np.moveaxis(
                np.moveaxis(velocity_grid, 0, 3), 4, 0
            )
            alpha_array = np.moveaxis(np.moveaxis(alpha_grid, 0, 3), 4, 0)

            # Fourth axis contains channels, i.e. three velocity components
            # and subsequently the alpha field
            grid = np.concatenate((velocity_array, alpha_array), axis=4)

            # Finally appending to created dataset
            if i == 0 and k == 0:
                grids = grid
            else:
                grids = np.concatenate((grids, grid), axis=0)

    np.save(out_file, grids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Module that wraps some \
legacy code to interpolate data from  an unstructured mesh to a structured \
mesh and calculate subgrid snapshots from output for 3D slug flow dataset.")
    parser.add_argument('--random', type=int, nargs='?',
                        default=1,
                        help='If 1, select random samples, otherwise split\
 grid 10-fold')
    parser.add_argument('--nfiles', type=int, nargs='?',
                        default=2,
                        help='Number of vtu files')
    parser.add_argument('--offset', type=int, nargs='?',
                        default=0,
                        help='vtu file to start from')
    parser.add_argument('--ndatapoints', type=int, nargs='?',
                        default=2,
                        help='number of random subdomains to sample from each vtu\
 file')
    parser.add_argument('--in_file_base', type=str, nargs='?',
                        default="slug_255_exp_projected_",
                        help='base filename for vtu files')
    parser.add_argument('--out_file', type=str, nargs='?',
                        default="sf_snapshots.npy",
                        help='output datafile')
    args = parser.parse_args()

    arg_dict = vars(args)

    get_snapshots_3D(**arg_dict)
