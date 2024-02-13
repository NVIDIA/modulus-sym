# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pathlib import Path
from modulus.sym.geometry import Parameterization, Parameter, Bounds
from modulus.sym.geometry.primitives_1d import Point1D, Line1D
from modulus.sym.geometry.primitives_2d import (
    Line,
    Channel2D,
    Rectangle,
    Circle,
    Triangle,
    Ellipse,
    Polygon,
)
from modulus.sym.geometry.primitives_3d import (
    Plane,
    Channel,
    Box,
    Sphere,
    Cylinder,
    Torus,
    Cone,
    TriangularPrism,
    Tetrahedron,
    IsoTriangularPrism,
    ElliCylinder,
)
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.utils.io.vtk import var_to_polyvtk

dir_path = Path(__file__).parent


def check_geometry(
    geo,
    criteria=None,
    parameterization=None,
    bounds=None,
    boundary_area=None,
    interior_area=None,
    max_sdf=None,
    compute_sdf_derivatives=True,
    check_bounds=None,
    debug=False,
):
    if debug:
        print("checking geo: " + str(geo))

    # check boundary
    if boundary_area is not None:
        boundary = geo.sample_boundary(
            1000, criteria=criteria, parameterization=parameterization
        )
        if debug:
            var_to_polyvtk(boundary, "boundary.vtp")
        assert np.isclose(np.sum(boundary["area"]), boundary_area, rtol=1e-1)

    # check interior
    if interior_area is not None:
        interior = geo.sample_interior(
            1000,
            criteria=criteria,
            parameterization=parameterization,
            bounds=bounds,
            compute_sdf_derivatives=compute_sdf_derivatives,
        )
        if debug:
            var_to_polyvtk(interior, "interior.vtp")
        assert np.isclose(np.sum(interior["area"]), interior_area, rtol=1e-1)

        if max_sdf is not None:
            assert np.max(interior["sdf"]) < max_sdf

        if compute_sdf_derivatives:
            sdf_diff = np.concatenate(
                [interior["sdf__" + d] for d in geo.dims], axis=-1
            )
            assert np.all(
                np.isclose(np.mean(np.linalg.norm(sdf_diff, axis=1)), 1.0, rtol=1e-1)
            )


def test_primitives():
    # point 1d
    g = Point1D(1)
    check_geometry(g, boundary_area=1)

    # line 1d
    g = Line1D(1, 2.5)
    check_geometry(g, boundary_area=2, interior_area=1.5, max_sdf=0.75)

    # line
    g = Line((1, 0), (1, 2.5), normal=1)
    check_geometry(g, boundary_area=2.5)

    # channel
    g = Channel2D((0, 0), (2, 3))
    check_geometry(g, boundary_area=4, interior_area=6, max_sdf=1.5)

    # rectangle
    g = Rectangle((0, 0), (2, 3))
    check_geometry(g, boundary_area=10, interior_area=6, max_sdf=1.0)

    # circle
    g = Circle((0, 2), 2)
    check_geometry(g, boundary_area=4 * np.pi, interior_area=4 * np.pi, max_sdf=2.0)

    # triangle
    g = Triangle((0, 0.5), 1, 1)
    check_geometry(
        g,
        boundary_area=1.0 + 2 * np.sqrt(0.5**2 + 1.0),
        interior_area=0.5,
        max_sdf=0.30897,
    )

    # ellipse
    g = Ellipse((0, 2), 1, 2)
    check_geometry(g, boundary_area=9.688448, interior_area=2 * np.pi, max_sdf=1.0)

    # polygon
    g = Polygon([(0, 0), (2, 0), (2, 1), (1, 2), (0, 1)])
    check_geometry(g, boundary_area=4 + 2 * np.sqrt(2), interior_area=3.0)

    # plane
    g = Plane((0, -1, 0), (0, 1, 2))
    check_geometry(g, boundary_area=4)

    # channel
    g = Channel((0, 0, -1), (2, 3, 4))
    check_geometry(g, boundary_area=32, interior_area=30, max_sdf=1.5)

    # box
    g = Box((0, 0, -1), (2, 3, 4))
    check_geometry(g, boundary_area=62, interior_area=30, max_sdf=1)

    # sphere
    g = Sphere((0, 1, 2), 2)
    check_geometry(g, boundary_area=16 * np.pi, interior_area=np.pi * 8 * 4 / 3.0)

    # cylinder
    g = Cylinder((0, 1, 2), 2, 3)
    check_geometry(g, boundary_area=20 * np.pi, interior_area=12 * np.pi, max_sdf=1.5)

    # torus
    g = Torus((0, 1, 2), 2, 1)
    check_geometry(
        g, boundary_area=8 * np.pi**2, interior_area=4 * np.pi**2, max_sdf=1
    )

    """
    # cone
    g = Cone((0, 1, 2), 1, 3)
    checks.append((g, np.pi*(1+np.sqrt(10)), np.pi, 0, None))

    # triangular prism
    g = TriangularPrism((0, 1, 2), 1, 2)
    checks.append((g, 6*np.sqrt(2) + 1, 2, 0, None))

    # tetrahedron
    g = Tetrahedron((0, 1, 2), 1)
    checks.append((g, np.sqrt(3), 1.0/(6.0*np.sqrt(2)), 0, None))
    """

    # box scale
    g = Box((0, 0, 0), (1, 2, 3))
    g = g.scale(2)
    check_geometry(g, boundary_area=88, interior_area=48, max_sdf=1)

    # box translate
    g = Box((0, 0, 0), (1, 2, 3))
    g = g.translate((0, 1, 2))
    check_geometry(g, boundary_area=22, interior_area=6, max_sdf=0.5)

    # box rotate
    g = Box((0, 0, 0), (1, 2, 3))
    g = g.rotate(np.pi / 4.0, axis="x", center=(10, -1, 20))
    g = g.rotate(np.pi / 4.0, axis="y")
    g = g.rotate(np.pi / 4.0, axis="z", center=(10, -10, 20))
    check_geometry(g, boundary_area=22, interior_area=6, max_sdf=0.5)

    # repeat operation
    g = Sphere((0, 0, 0), 0.5)
    g = g.repeat(1.5, [-1, -1, -1], [3, 3, 3])
    check_geometry(
        g,
        boundary_area=np.pi * 5**3,
        interior_area=(1.0 / 6.0) * np.pi * 5**3,
        max_sdf=0.5,
    )

    # tessellated geometry
    g = Tessellation.from_stl(dir_path / "stls/cube.stl")
    check_geometry(g, boundary_area=6, interior_area=1.0, max_sdf=0.5)

    # tessellated with primitives geometry
    g = Tessellation.from_stl(dir_path / "stls/cube.stl") - Box(
        (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
    )
    check_geometry(g, boundary_area=6, interior_area=0.875)

    # Integral plane
    sdf_fn = Tessellation.from_stl(dir_path / "stls/cube.stl") - Box(
        (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
    )

    def _interior_criteria(sdf_fn):
        def interior_criteria(invar, params):
            sdf = sdf_fn.sdf(invar, params)
            return np.greater(sdf["sdf"], 0)

        return interior_criteria

    g = Plane((0.25, 0, 0), (0.25, 1, 1))
    check_geometry(g, boundary_area=0.75, criteria=_interior_criteria(sdf_fn))

    # test parameterization
    radius = Parameter("radius")
    angle = Parameter("angle")
    g = Circle((0, 0, 0), radius, parameterization=Parameterization({radius: (1, 2)}))
    g = Rectangle((-2, -2, -2), (2, 2, 2)) - g
    g = g.rotate(
        angle=angle, parameterization=Parameterization({angle: (0, 2.0 * np.pi)})
    )
    check_geometry(g, boundary_area=16 + 3 * np.pi)
    check_geometry(
        g,
        boundary_area=16 + 2 * np.pi,
        parameterization=Parameterization({radius: 1, angle: np.pi}),
    )


test_primitives()
