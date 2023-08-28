from collections import defaultdict
from itertools import chain
from cctbx.array_family import flex
from cctbx import crystal
import cctbx
import math
import numpy as np
from scipy.spatial.distance import cdist
from scitbx_array_family_flex_ext import vec3_double, double, size_t

def flatten_and_shape(lst):
    """Flatten a nested list and return its shape."""
    def helper(l):
        if not isinstance(l, list):
            return [l], ()
        flat = []
        shapes = []
        for item in l:
            f, s = helper(item)
            flat.extend(f)
            shapes.append(s)
        if len(set(shapes)) != 1:
            raise ValueError("Ragged nested list detected.")
        return flat, (len(l),) + shapes[0]

    flattened, shape = helper(lst)
    return flattened, shape

def get_dtype_of_list(lst):
    dtypes = {type(item) for item in lst}
    
    if len(dtypes) > 1:
        raise ValueError("Multiple data types detected.")
    elif len(dtypes) == 0:
        raise ValueError("Empty list provided.")
    else:
        return dtypes.pop()
    
def flex_from_list(lst,signed_int=False):
    flat_list, shape = flatten_and_shape(lst)
    dtype = get_dtype_of_list(flat_list)
    type_mapper = {int:flex.size_t,
                   float:flex.double,
                   bool:flex.bool}
    if signed_int:
        type_mapper[int] = flex.int16
    
    # make flex array
    assert dtype in type_mapper, f"Unrecognized type: {dtype}"
    flex_func = type_mapper[dtype]
    flex_array = flex_func(flat_list)
    if len(shape)>1:
        flex_array.reshape(flex.grid(*shape))
    return flex_array

def nd_to_1d_indices(indices, shape):
    # Normalize indices to always use slice objects
    normalized_indices = []
    for dim, idx in enumerate(indices):
        if idx is None:
            normalized_indices.append(slice(0,shape[dim]))
        else:
            normalized_indices.append(idx)
    
    # If any index is a slice, recursively call function for each value in slice
    for dim, (i, s) in enumerate(zip(normalized_indices, shape)):
        if isinstance(i, slice):
            result_indices = []
            start, stop, step = i.indices(s)
            for j in range(start, stop, step):
                new_indices = list(normalized_indices)
                new_indices[dim] = j
                result_indices.extend(nd_to_1d_indices(new_indices, shape))
            return result_indices
    
    # If no slices, calculate single 1D index
    index = 0
    stride = 1
    for i, dim in reversed(list(zip(normalized_indices, shape))):
        index += i * stride
        stride *= dim
    return [index]

def optimized_nd_to_1d_indices(i, shape):
    # For fixed input of (None, i, None), we directly compute based on given structure
    result_indices = []
    
    # Pre-compute for 1st dimension which is always a slice
    start1, stop1 = 0, shape[0]
    
    # Pre-compute for 3rd dimension which is always a slice
    start3, stop3 = 0, shape[2]
    stride3 = 1
    
    # Directly compute for 2nd dimension which is variable
    stride2 = shape[2]
    index2 = i * stride2 * shape[0]
    
    for val1 in range(start1, stop1):
        for val3 in range(start3, stop3):
            result_indices.append(val1 * stride2 + index2 + val3 * stride3)
            
    return result_indices



# def query_ball_point_flex(points, query_point, radius, return_length=True):
#     """
#     For a given query point, find all points within a given radius.

#     Parameters:
#     - points (np.array): Set of 3D points of shape (N, 3).
#     - query_point (np.array): The 3D query point of shape (1, 3).
#     - radius (float): The query radius.
#     - return_length (bool, optional): If True, return only the count of neighbors.

#     Returns:
#     - indices (list): If return_length=False, indices of points within the radius. Else, count of such points.
#     """
    
#     # Calculate the distance from all points to the query point
#     distances = cdist_flex(points, query_point).as_1d()
    
#     # Find indices of points within the radius
#     sel = distances < radius
#     indices = flex.uint32_range(points.focus()[0])
#     indices = indices.select(sel)
#     if return_length:
#         return len(indices)
#     else:
#         return indices

def cdist_flex(A,B):
    

    def indices_2d_flex(dimensions):
        N = len(dimensions)
        if N != 2:
            raise ValueError("Only 2D is supported for this implementation.")

        # Create the row indices
        row_idx = flex.size_t(chain.from_iterable([[i] * dimensions[1] for i in range(dimensions[0])]))

        # Create the column indices
        col_idx = flex.size_t(chain.from_iterable([list(range(dimensions[1])) for _ in range(dimensions[0])]))

        return row_idx, col_idx

        
    i_idxs, j_idxs = indices_2d_flex((A.focus()[0],B.focus()[0]))
    
    r = i_idxs
    xi = i_idxs*3
    yi = i_idxs*3 + 1
    zi = i_idxs*3 + 2

    xa = A.select(xi)
    ya = A.select(yi)
    za = A.select(zi)

    xj = j_idxs*3
    yj = j_idxs*3 + 1
    zj = j_idxs*3 + 2


    xb = B.select(xj)
    yb = B.select(yj)
    zb = B.select(zj)

    d = ((xb - xa)**2 + (yb - ya)**2 + (zb - za)**2)**0.5
    d.reshape(flex.grid((A.focus()[0],B.focus()[0])))
    
    return d

def sphere_points_flex(ctr, rad, N):
    if ctr.ndim==1:
        ctr = ctr[None,:]
    h = -1.0 + (2.0 * np.arange(N) / float(N-1))[:, np.newaxis]
    phis = np.arccos(h)
    thetas = np.zeros_like(phis)
    thetas[1:-1, :] = (3.6 / np.sqrt(N * (1.0 - h[1:-1]**2))) % (2 * np.pi)
    thetas = np.cumsum(thetas, axis=0)

    x = np.sin(phis) * np.cos(thetas)
    y = np.sin(phis) * np.sin(thetas)
    z = np.cos(phis)

    # Stack x, y, z to form points and multiply by rad
    points = rad * np.stack([x, y, z], axis=-1)

    # Reshape points to (1, N, 3)
    points = points.reshape(1, N, 3)

    # Add center coordinates to all points
    # ctr shape: (M, 3), points shape: (1, N, 3)
    # Resultant shape: (M, N, 3)
    pts = ctr[:, np.newaxis, :] + points


    return pts

def query_atom_neighbors(model,radius=3.5,include_self=True,only_unit=True):
    crystal_symmetry = model.crystal_symmetry()
    hierarchy = model.get_hierarchy()
    sites_cart = hierarchy.atoms().extract_xyz()
    sst = crystal_symmetry.special_position_settings().site_symmetry_table(
    sites_cart = sites_cart)
    conn_asu_mappings = crystal_symmetry.special_position_settings().\
    asu_mappings(buffer_thickness=5)
    conn_asu_mappings.process_sites_cart(
    original_sites      = sites_cart,
    site_symmetry_table = sst)
    conn_pair_asu_table = cctbx.crystal.pair_asu_table(
    asu_mappings=conn_asu_mappings)
    conn_pair_asu_table.add_all_pairs(distance_cutoff=radius)
    pair_generator = cctbx.crystal.neighbors_fast_pair_generator(
    conn_asu_mappings,
    distance_cutoff=radius)
    fm = crystal_symmetry.unit_cell().fractionalization_matrix()
    om = crystal_symmetry.unit_cell().orthogonalization_matrix()

    def dist_expl(r1, r2, op):
        r1_f = fm*flex.vec3_double([r1]) # convert to fractional coordinates
        r1_f_mapped = op*r1_f[0] # apply symmetry operator
        r1_c_mapped = (om*flex.vec3_double([r1_f_mapped]))[0] # back to Cartesian
        return math.sqrt( (r1_c_mapped[0]-r2[0])**2 +
                          (r1_c_mapped[1]-r2[1])**2 +
                          (r1_c_mapped[2]-r2[2])**2 )

    pairs = list(pair_generator)
    inds = defaultdict(list)
    dists = defaultdict(list)
    
    for pair in pairs:
        i,j = pair.i_seq, pair.j_seq
        rt_mx_i = conn_asu_mappings.get_rt_mx_i(pair)
        rt_mx_j = conn_asu_mappings.get_rt_mx_j(pair)
        rt_mx_ji = rt_mx_i.inverse().multiply(rt_mx_j)

                
        if (only_unit and rt_mx_ji.is_unit_mx()) or (not only_unit):
            d = round(math.sqrt(pair.dist_sq),6)
            inds[i].append(j)
            dists[i].append(d)
            
            # add reverse
            inds[j].append(i)
            dists[j].append(d)
            #print(pair.i_seq, pair.j_seq, rt_mx_ji, math.sqrt(pair.dist_sq), de)
    
    # add self
    if include_self:
        for key,value in list(inds.items()):
            dval = dists[key]
            dists[key]= dval+[0.0]
            inds[key] = value+[key]

    # sort
    for key,value in list(inds.items()):
        dval = dists[key]
        # sort
        sorted_pairs = sorted(set(list(zip(value,dval))))
        value_sorted, dval_sorted = zip(*sorted_pairs)
        inds[key] = flex.size_t(value_sorted)
        dists[key] = flex.double(dval_sorted)


    return inds,dists

def sphere_points_cctbx(ctr, rad, N):
    ctr = np.array(ctr)
    if ctr.ndim==1:
        ctr = ctr[None,:]
    h = -1.0 + (2.0 * np.arange(N) / float(N-1))[:, np.newaxis]
    phis = np.arccos(h)
    thetas = np.zeros_like(phis)
    thetas[1:-1, :] = (3.6 / np.sqrt(N * (1.0 - h[1:-1]**2))) % (2 * np.pi)
    thetas = np.cumsum(thetas, axis=0)

    x = np.sin(phis) * np.cos(thetas)
    y = np.sin(phis) * np.sin(thetas)
    z = np.cos(phis)

    # Stack x, y, z to form points and multiply by rad
    points = rad * np.stack([x, y, z], axis=-1)

    # Reshape points to (1, N, 3)
    points = points.reshape(1, N, 3)

    # Add center coordinates to all points
    # ctr shape: (M, 3), points shape: (1, N, 3)
    # Resultant shape: (M, N, 3)
    pts = ctr[:, np.newaxis, :] + points


    return pts


def query_ball_point_flex(tree,tree_xyz,query_xyz,r=1.0):
    n_atoms,n_probes, _ = query_xyz.focus()
    counts = []

    for atom_i in range(n_atoms):
        probe_range = (n_probes * atom_i * 3, n_probes * (atom_i+1) * 3)
        atom_probes_xyz = query_xyz.select(flex.size_t_range(*probe_range))
        atom_probes_xyz.reshape(flex.grid(n_probes,3))
        nbrs = tree[atom_i]
        n_nbrs = len(nbrs)
        nbrs_xyz = tree_xyz.select(flex.size_t(nbrs)).as_1d().as_double()
        nbrs_xyz.reshape(flex.grid(len(nbrs),3))
        d = cdist_flex(nbrs_xyz,atom_probes_xyz)
        sel = d<r
        count = []
        for nbr_i in range(n_probes):
            nbr_range = (slice(0,n_nbrs),slice(nbr_i,nbr_i+1))
            count_nbr = sel[nbr_range].count(True)
            count.append(count_nbr)

        counts.append(count)

    counts = flex_from_list(counts)
    return counts

def radial_shell_worker_cctbx(args):

    i,model,n_probes,n_probes_target,radius_shell,rtol = args
    atoms_xyz = model.get_sites_cart()
    n_atoms = len(atoms_xyz)
    numPts = n_probes_target
    RAD = radius_shell
    outRAD = rtol
    inds,dists = query_atom_neighbors(model)

    probe_xyz = flex.double(n_atoms*n_probes_target*3,-1.0) # init flat xyz array
    probe_xyz.reshape(flex.grid(n_atoms,n_probes_target,3))
    #probe_xyz = FlexContainer(probe_xyz,shape=(n_atoms,n_probes,3))

    keep_mask = flex.bool(n_atoms*n_probes_target,False)
    keep_mask.reshape(flex.grid(n_atoms,n_probes_target))
    #keep_mask = FlexContainer(keep_mask,shape=(n_atoms,n_probes))
    all_pts = []
    for atom_i,_ in enumerate(range(7)):
        coord = atoms_xyz[atom_i]
        print("coord:",coord)
        pts = []

        # try to get at least [numPts] points at [RAD] distance
        # from the atom, that are not closer to other atoms
        for i in range (0, 50) :
            # if we find the necessary number of probes in the first iteration, then i will never go to 1
            # points on a sphere at radius RAD...
            n_pts_to_grab = numPts+i*2 # progressively more points are grabbed  with each failed iter
            print("n_to_grab",n_pts_to_grab)
            #print("n_to_grab:",n_pts_to_grab)
            outPts = sphere_points_cctbx(coord,RAD,n_pts_to_grab) # get the points
            outPts = flex.double(outPts[0])
            #outPts = [outPts[i:i+1] for i in range(len(outPts))]
            at_pts, at_pts_i = [None]*len(outPts), 0
            print("candidate probes:")
                
            for pt_i in range(n_pts_to_grab): # identify which ones to keep, progressively grow pts list
                pt = outPts.select([pt_i*3,pt_i*3 + 1, pt_i * 3 +2])
                print(f"\t{pt[0]},{pt[1]},{pt[2]}")
                nbrs = inds[atom_i]
                nbrs_xyz = atoms_xyz.select(nbrs)
                nbrs_xyz_double = nbrs_xyz.as_double()
                nbrs_xyz_double.reshape(flex.grid(len(nbrs_xyz),3))
                pt.reshape(flex.grid(1,3))
                d = cdist_flex(nbrs_xyz_double,pt)
                count = (d < RAD*outRAD).count(True)
                # each value in counts is the number of atoms within radius+tol of each probe
                ptsNear = count

                if ptsNear == 0 :
                    at_pts[at_pts_i] = pt
                    at_pts_i += 1
                # if at_pts_i >= numPts:
                #     break

            if at_pts_i >= numPts : # if we have enough points, take all the "good" points from this iter
                pts.extend ( at_pts[0:at_pts_i] )
                break
        # debug append to list
        all_pts.append(pts)
        
        #add points to array for each atom
        for j,pt in enumerate(pts):
            if j<n_probes_target:
                x,y,z = pt
                probe_xyz[atom_i,j,0] = pt[0]
                probe_xyz[atom_i,j,1] = pt[1]
                probe_xyz[atom_i,j,2] = pt[2]
                keep_mask[atom_i,j] = True
                
    return probe_xyz, keep_mask, all_pts