import math
import random
import itertools
import numpy as np
from scipy.spatial import cKDTree

from tqdm import tqdm
import warnings

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import matplotlib.offsetbox
import matplotlib.cbook


def circles(x, y, s, c="b", vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault("color", c)
        c = None
    if "fc" in kwargs:
        kwargs.setdefault("facecolor", kwargs.pop("fc"))
    if "ec" in kwargs:
        kwargs.setdefault("edgecolor", kwargs.pop("ec"))
    if "ls" in kwargs:
        kwargs.setdefault("linestyle", kwargs.pop("ls"))
    if "lw" in kwargs:
        kwargs.setdefault("linewidth", kwargs.pop("lw"))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection


def arc_patch(
    center, radius, theta1, theta2, ax=None, resolution=100, closed=False, **kwargs
):
    # make sure ax is not empty
    if ax is None:
        ax = plt.gca()
    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack(
        (radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1])
    )
    # build the polygon and add it to the axes
    poly = Polygon(points.T, closed=closed, **kwargs)
    ax.add_patch(poly)
    return poly


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """
    size: length of bar in data units
    extent : height of bar ends in axes units
    """

    def __init__(
        self,
        size=1,
        extent=0.03,
        label="",
        loc=2,
        ax=None,
        pad=0.4,
        borderpad=0.5,
        ppad=0,
        sep=2,
        prop=None,
        frameon=True,
        textkw={},
        linekw={},
        **kwargs,
    ):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **linekw)
        vline1 = Line2D([0, 0], [-extent / 2.0, extent / 2.0], **linekw)
        vline2 = Line2D([size, size], [-extent / 2.0, extent / 2.0], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(
            label, minimumdescent=False, textprops=textkw
        )
        self.vpac = matplotlib.offsetbox.VPacker(
            children=[size_bar, txt], align="center", pad=ppad, sep=sep
        )
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=self.vpac,
            prop=prop,
            frameon=frameon,
            **kwargs,
        )


def euclidean_distance(x1, y1, x2, y2):
    return math.hypot((x1 - x2), (y1 - y2))


def minimise_distance(center, points_coordinates):
    points_coordinates -= center
    distances = np.sqrt(points_coordinates[:, 0] ** 2 + points_coordinates[:, 1] ** 2)
    return np.argmin(distances)


def minimise_arclength(particle_radius, particle_theta, ray_theta):
    return np.argmin(particle_radius * np.abs(particle_theta - ray_theta))


# Configure canvas and black hole positions
canvas_boundaries = np.array([[0, 4], [-1, 4]])
number_particles = 130

# Include the black holes to avoid overlap with the gas particles
circle_list = [(1, 1, 0.05), (1, 3, 0.05), (3, 1, 0.05), (3, 3, 0.05)]

# Set the random number generator to a fixed seed
random.seed(25)  # The paper adopts seed = 25

# Initialise matplotlib canvas
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
plt.style.use("mnras.mplstyle")
fig = plt.figure(figsize=(4 * 0.65, 5 * 0.65), constrained_layout=True)
ax = fig.subplots()
ax.set_aspect("equal")
ax.set_xlim(canvas_boundaries[0])
ax.set_ylim(canvas_boundaries[1])
ax.axis("off")

# Create labels for heating schemes
title_kwargs = dict(
    color="k",
    ha="center",
    va="top",
    alpha=1,
    transform=ax.transData,
    bbox=dict(facecolor="none", edgecolor="none", alpha=0.8, zorder=0),
)
ax.text(1, 3.95, "Minimum distance", **title_kwargs)
ax.text(3, 3.95, "Random", **title_kwargs)
ax.text(1, 1.95, "Isotropic", **title_kwargs)
ax.text(3, 1.95, "Bipolar", **title_kwargs)

# Plot black holes
black_hole_coordinates = np.array([[1, 1], [1, 3], [3, 1], [3, 3]], dtype=np.float32)
black_hole_radius = 0.08
black_holes = np.hstack(
    (
        black_hole_coordinates,
        np.ones(len(black_hole_coordinates))[:, None] * black_hole_radius,
    )
)
circles(*black_holes.T, alpha=1, color="k", edgecolor="none", zorder=10)

# Plot black hole kernels
black_hole_kernel_radius = 0.75
black_holes_kernel = np.hstack(
    (
        black_hole_coordinates,
        np.ones(len(black_hole_coordinates))[:, None] * black_hole_kernel_radius,
    )
)
circles(
    black_holes_kernel[:, 0],
    black_holes_kernel[:, 1],
    black_holes_kernel[:, 2] / 2,
    alpha=0.5,
    color="none",
    edgecolor="green",
    zorder=1,
)
circles(*black_holes_kernel.T, alpha=0.25, color="lime", edgecolor="none", zorder=1)

# Generate gas particles
number_attempts = 0
with tqdm(desc="Generating gas particles", total=number_particles) as progress_bar:

    while len(circle_list) < number_particles + 4:

        r = random.uniform(5.5, 7) * 1e-3 * math.hypot(*canvas_boundaries[:, 1])
        x = random.uniform(canvas_boundaries[0, 0] - r, canvas_boundaries[0, 1] - r)
        y = random.uniform(canvas_boundaries[1, 0] - r, canvas_boundaries[1, 1] - r)
        number_attempts += 1
        # Loop over points to check for overlap
        if not any(
            (x2, y2, r2)
            for x2, y2, r2 in circle_list
            if euclidean_distance(x, y, x2, y2) < r + r2
        ):
            circle_list.append((x, y, r))
            progress_bar.update(1)

print(
    f"Generated {number_particles:d} non-overlapping gas particles in {number_attempts:d} attempts."
)

# Clip out the black holes from the gas list
circle_list = np.asarray(circle_list)[4:]
gas_coordinates = circle_list[:, :2]

# Find gas particles within the black hole search kernels
point_tree = cKDTree(gas_coordinates)
search_index = point_tree.query_ball_point(
    black_hole_coordinates, black_hole_kernel_radius
)
search_index = np.asarray(list(itertools.chain.from_iterable(search_index)))
inverse_search_index = np.ones(number_particles, dtype=bool)
inverse_search_index[search_index] = False

circles(
    *circle_list[search_index].T,
    alpha=1,
    color="orange",
    edgecolor="w",
    linewidth=0.3,
    zorder=9,
)
circles(
    *circle_list[inverse_search_index].T,
    alpha=0.6,
    color="none",
    edgecolor="grey",
    linewidth=0.5,
    zorder=1,
)

#
# >>> PARTICLE SELECTION METHODS <<<
#
# Minimum distance
bh_index = 1
search_index_random = point_tree.query_ball_point(
    black_hole_coordinates[bh_index], black_hole_kernel_radius
)
candidate_gas_coordinates = point_tree.data[search_index_random]
particle_min_arclength_index = minimise_distance(
    black_hole_coordinates[bh_index], candidate_gas_coordinates
)
index_neighbour = search_index_random[particle_min_arclength_index]

plt.plot(
    [black_hole_coordinates[bh_index, 0], circle_list[index_neighbour, 0]],
    [black_hole_coordinates[bh_index, 1], circle_list[index_neighbour, 1]],
    linewidth=1,
)
circles(
    *circle_list[index_neighbour].T,
    alpha=1,
    color="red",
    edgecolor="w",
    linewidth=0.3,
    zorder=9,
)

# Random
bh_index = 3
search_index_random = point_tree.query_ball_point(
    black_hole_coordinates[bh_index], black_hole_kernel_radius
)
search_index_random = random.choice(search_index_random)
circles(
    *circle_list[search_index_random].T,
    alpha=1,
    color="red",
    edgecolor="w",
    linewidth=0.3,
    zorder=9,
)

# Isotropic
bh_index = 0
search_index_kernel = point_tree.query_ball_point(
    black_hole_coordinates[bh_index], black_hole_kernel_radius
)
gas_coordinates_kernel_scaled = (
    point_tree.data[search_index_kernel] - black_hole_coordinates[bh_index]
)
particle_theta = np.arctan2(
    gas_coordinates_kernel_scaled[:, 1], gas_coordinates_kernel_scaled[:, 0]
)
particle_radial_distance = np.sqrt(
    gas_coordinates_kernel_scaled[:, 1] ** 2 + gas_coordinates_kernel_scaled[:, 0] ** 2
)

ray_theta = 170 / 180 * np.pi
plt.plot(
    [
        black_hole_coordinates[bh_index, 0],
        black_hole_coordinates[bh_index, 0]
        + black_hole_kernel_radius * 1.2 * np.cos(ray_theta),
    ],
    [
        black_hole_coordinates[bh_index, 1],
        black_hole_coordinates[bh_index, 1]
        + black_hole_kernel_radius * 1.2 * np.sin(ray_theta),
    ],
    linewidth=1,
    color="blue",
)

particle_min_arclength_index = minimise_arclength(
    particle_radial_distance, particle_theta, ray_theta
)
search_index_isotropic = search_index_kernel[particle_min_arclength_index]
circles(
    *circle_list[search_index_isotropic].T,
    alpha=1,
    color="red",
    edgecolor="w",
    linewidth=0.3,
    zorder=9,
)
arc_patch(
    black_hole_coordinates[bh_index],
    particle_radial_distance[particle_min_arclength_index],
    math.degrees(ray_theta),
    math.degrees(particle_theta[particle_min_arclength_index]),
    ax=ax,
    fill=False,
    color="blue",
)

# Bipolar
bh_index = 2
search_index_kernel = point_tree.query_ball_point(
    black_hole_coordinates[bh_index], black_hole_kernel_radius
)
gas_coordinates_kernel_scaled = (
    point_tree.data[search_index_kernel] - black_hole_coordinates[bh_index]
)
particle_theta = np.arctan2(
    gas_coordinates_kernel_scaled[:, 1], gas_coordinates_kernel_scaled[:, 0]
)
particle_radial_distance = np.sqrt(
    gas_coordinates_kernel_scaled[:, 1] ** 2 + gas_coordinates_kernel_scaled[:, 0] ** 2
)

ray_theta = 0
plt.plot(
    [
        black_hole_coordinates[bh_index, 0],
        black_hole_coordinates[bh_index, 0]
        + black_hole_kernel_radius * 1.2 * np.cos(ray_theta),
    ],
    [
        black_hole_coordinates[bh_index, 1],
        black_hole_coordinates[bh_index, 1]
        + black_hole_kernel_radius * 1.2 * np.sin(ray_theta),
    ],
    linewidth=1,
    color="blue",
)

particle_min_arclength_index = minimise_arclength(
    particle_radial_distance, particle_theta, ray_theta
)
search_index_isotropic = search_index_kernel[particle_min_arclength_index]
circles(
    *circle_list[search_index_isotropic].T,
    alpha=1,
    color="red",
    edgecolor="w",
    linewidth=0.3,
    zorder=9,
)

arc_patch(
    black_hole_coordinates[bh_index],
    particle_radial_distance[particle_min_arclength_index],
    math.degrees(ray_theta),
    math.degrees(particle_theta[particle_min_arclength_index]),
    ax=ax,
    fill=False,
    color="blue",
)

ray_theta = np.pi
plt.plot(
    [
        black_hole_coordinates[bh_index, 0],
        black_hole_coordinates[bh_index, 0]
        + black_hole_kernel_radius * 1.2 * np.cos(ray_theta),
    ],
    [
        black_hole_coordinates[bh_index, 1],
        black_hole_coordinates[bh_index, 1]
        + black_hole_kernel_radius * 1.2 * np.sin(ray_theta),
    ],
    linestyle="--",
    linewidth=1,
    color="blue",
)

particle_min_arclength_index = minimise_arclength(
    particle_radial_distance, particle_theta, ray_theta
)
search_index_isotropic = search_index_kernel[particle_min_arclength_index]
circles(
    *circle_list[search_index_isotropic].T,
    alpha=1,
    color="none",
    edgecolor="red",
    linestyle="-",
    linewidth=0.5,
    zorder=9,
)

arc_patch(
    black_hole_coordinates[bh_index],
    particle_radial_distance[particle_min_arclength_index],
    math.degrees(ray_theta),
    math.degrees(particle_theta[particle_min_arclength_index]),
    ax=ax,
    fill=False,
    color="blue",
    linewidth=1,
    linestyle="--",
)

# Display the scale of the image
scale_box = AnchoredHScaleBar(
    size=1,
    label="$\\approx$ 1 kpc",
    loc="lower right",
    frameon=False,
    pad=0.6,
    sep=4,
    extent=0.0,
    linekw=dict(color="k", linewidth=1.25),
    textkw=dict(color="k"),
)
ax.add_artist(scale_box)

bh_coords = black_hole_coordinates[3]
ax.annotate(
    text="BH search radius",
    xy=(
        bh_coords[0] + black_hole_kernel_radius / np.sqrt(2),
        bh_coords[1] - black_hole_kernel_radius / np.sqrt(2),
    ),
    xycoords="data",
    fontsize=6,
    xytext=(15, -15),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="simple",
        facecolor="black",
        edgecolor="none",
        shrinkA=1,
        shrinkB=0.7,
        mutation_scale=5,
    ),
    horizontalalignment="center",
    verticalalignment="center",
)

ax.annotate(
    text="SPH smoothing radius",
    xy=(
        bh_coords[0] - black_hole_kernel_radius / 2 / np.sqrt(2),
        bh_coords[1] - black_hole_kernel_radius / 2 / np.sqrt(2),
    ),
    xycoords="data",
    fontsize=6,
    xytext=(-15, -20),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="simple",
        facecolor="black",
        edgecolor="none",
        shrinkA=1,
        shrinkB=0.7,
        mutation_scale=5,
    ),
    horizontalalignment="center",
    verticalalignment="center",
)

bh_coords = black_hole_coordinates[2]
ax.annotate(
    text="$+\\overrightarrow{{x}}$",
    xy=(
        bh_coords[0] + black_hole_kernel_radius * 1.25,
        bh_coords[1],
    ),
    xycoords="data",
    fontsize=5,
    horizontalalignment="left",
    verticalalignment="center",  # Center vertically
)
ax.annotate(
    text="$-\\overrightarrow{{x}}$",
    xy=(
        bh_coords[0] - black_hole_kernel_radius * 1.25,
        bh_coords[1],
    ),
    xycoords="data",
    fontsize=5,
    horizontalalignment="right",
    verticalalignment="center",  # Center vertically
)

# Add legend
handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Heated gas",
        markerfacecolor="r",
        markersize=7,
        linewidth=0,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Gas in search sphere",
        markerfacecolor="orange",
        markersize=7,
        linewidth=0,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        markeredgecolor="grey",
        label="Gas in field",
        markerfacecolor="none",
        markersize=6,
        linewidth=0,
        alpha=0.75,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="k",
        label="Black hole",
        markerfacecolor="k",
        markersize=6,
        linewidth=0,
    ),
]
ax.legend(
    handles=handles,
    loc="lower left",
    frameon=True,
    facecolor="w",
    edgecolor="none",
)
plt.margins(0, 0)
fig.savefig("heating_schemes.pdf", dpi=400)
plt.show()
