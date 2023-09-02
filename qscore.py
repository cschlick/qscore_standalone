import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Pool,cpu_count
from qscore_utils import (
          sphere_points,
          trilinear_interpolation,
          rowwise_corrcoef,
          )

from tqdm.notebook import tqdm

# def stack_fill(arrs, dim=1, max_to_shape=None, fill=-1.0):
#     max_shape = list(max(arr.shape for arr in arrs))
#     if max_to_shape is not None:
#         max_shape[np.argmax(max_shape)] = max_to_shape
    
#     ret_arr = np.full((len(arrs),) + tuple(max_shape), fill)
    
#     for i, arr in enumerate(arrs):
#         # Slice of indices to target the correct dimensions
#         indices = [slice(None)] * arr.ndim
#         indices[dim] = slice(arr.shape[dim])
#         if len(arr)>0:
#             ret_arr[i][tuple(indices)] = arr
    
#     return ret_arr

def stack_fill(arrs, max_to_shape,fill=-1.0):
    # Determine the shape of the output array based on max_to_shape and the first array in arrs
    result_shape = [len(arrs), max_to_shape] + list(arrs[0].shape[1:])
    result = np.full(result_shape, fill)

    for idx, arr in enumerate(arrs):
        # Decide the slice length based on max_to_shape and the current array shape
        slice_len = min(arr.shape[0], max_to_shape)

        # Create an index tuple to handle slicing in multiple dimensions
        index_tuple = (idx, slice(0, slice_len)) + (Ellipsis,)
        result[index_tuple] = arr[:slice_len]

    return result

def radial_shell_worker(args):
    i,atoms_xyz,n_probes,n_probes_target,radius_shell,rtol = args
    if radius_shell == 0:
        radius_shell = 1e-9 # zero causes crash
    numPts = n_probes_target
    RAD = radius_shell
    outRAD = rtol
    kdtree_atoms = KDTree(atoms_xyz)
    all_pts = [] # list of probe arrays for each atom
    probe_xyz_r = np.full((n_atoms,n_probes_target,3),-1.0)
    for atom_i,_ in enumerate(range(7)):
        coord = atoms_xyz[atom_i,:]
        print("coord:",coord)
        pts = []
        
        # try to get at least [numPts] points at [RAD] distance
        # from the atom, that are not closer to other atoms
        for i in range (0, 50) :
            # if we find the necessary number of probes in the first iteration, then i will never go to 1
            # points on a sphere at radius RAD...
            n_pts_to_grab = numPts+i*2 # progressively more points are grabbed  with each failed iter
            print("n_to_grab:",n_pts_to_grab)
            outPts = sphere_points(coord[None,:],RAD,n_pts_to_grab) # get the points
            outPts = outPts.reshape(-1, 3)
            at_pts, at_pts_i = [None]*len(outPts), 0
            print("probe candidates")
                
            for pt_i,pt in enumerate(outPts) : # identify which ones to keep, progressively grow pts list
                print(f"\t{pt[0]},{pt[1]},{pt[2]}")
                # query kdtree to find probe-atom interactions
                counts = kdtree_atoms.query_ball_point(pt[None,:],RAD*outRAD,return_length=True)
                
                # each value in counts is the number of atoms within radius+tol of each probe
                count = counts.flatten()[0]
                ptsNear = count
            
                if ptsNear == 0 :
                    at_pts[at_pts_i] = pt
                    at_pts_i += 1
                # if at_pts_i >= numPts:
                #     break
            
            if at_pts_i >= numPts : # if we have enough points, take all the "good" points from this iter
                pts.extend ( at_pts[0:at_pts_i] )
                break
        #assert len(pts)>0, "Zero probes were found "
        pts = np.array(pts) # should be shape (n_probes,3)
        all_pts.append(pts)
        
    # prepare output
    n_atoms = len(atoms_xyz)
    
    for i,r in enumerate(all_pts):
        probe_xyz_r[i,:n_probes,:] = r[:n_probes_target,:]

    keep_sel = probe_xyz_r != -1.
    keep_sel = np.mean(keep_sel, axis=-1, keepdims=True)  
    keep_sel = np.squeeze(keep_sel, axis=-1)

    return probe_xyz_r, keep_sel.astype(bool)


def radial_shell_mp(atoms_xyz,n_probes=64,n_probes_target=8,radii=None,rtol=0.9,num_processes=cpu_count()):



    # Create argument tuples for each chunk
    args = [(i,atoms_xyz,n_probes,n_probes_target,radius_shell,rtol) for i,radius_shell in enumerate(radii)]

    # Create a pool of worker processes
    if num_processes >1:
      with Pool(num_processes) as p:
          # Use the pool to run the trilinear_interpolation_worker function in parallel
          results = p.map(radial_shell_worker, args)
    else:
      results = []
      for arg in args:
        result = radial_shell_worker(arg)
        results.append(result)
    
    probe_xyz_all = [result[0] for result in results]
    keep_mask_all = [result[1] for result in results]
    all_pts_all = [result[2] for result in results]
    return all_pts_all # debug
    n_shells = len(radii)
    n_atoms = atoms_xyz.shape[0]
    out_shape = (n_shells,n_atoms,n_probes,3 )
    out_size = np.prod(out_shape)
    shell_size = np.prod(out_shape[1:])
    out_probes = np.full((n_shells,n_atoms,n_probes_target,3),-1.0)
    for i,p in enumerate(probe_xyz_all):
        out_probes[i,:,:p.shape[1],:] =p

    out_mask = np.full((n_shells,n_atoms,n_probes_target),False)
    for i,k in enumerate(out_mask):
        start = i*shell_size
        stop = start+shell_size
        out_mask[i] = k[:,:p.shape[1]]
    return out_probes,out_mask



def Qscore(volume,
                atoms_xyz,
                mask_clash=True,
                voxel_size=1.0,
                n_probes=8,
                min_probes=1,
                radii=np.arange(0.1,2.1,0.1),
                rtol=0.9,
                ignore_min_probes=False,
                selection_bool=None,
                num_processes=cpu_count()):

    # handle selection at the very beginning
    if selection_bool is None:
      selection_bool = np.full(atoms_xyz.shape[0],True)
    atoms_xyz = atoms_xyz[selection_bool]


              
    probe_xyz,keep_mask = radial_shell_mp(atoms_xyz,
                                          n_probes=n_probes,
                                          radii=radii,
                                          rtol = rtol,
                                          num_processes=num_processes)
    
    n_shells,n_atoms,n_probes,_ = probe_xyz.shape
    probe_xyz_flat = probe_xyz.reshape((n_atoms*n_shells*n_probes,3))
    keep_mask_flat = keep_mask.reshape(-1) # (n_shells*n_atoms*n_probes,)
    
    # apply mask to the flattened probe_xyz
    masked_probe_xyz_flat = probe_xyz_flat[keep_mask_flat]
    #masked_probe_xyz_flat_flex = flex.vec3_double(masked_probe_xyz_flat)
    
    # apply trilinear interpolation only to the relevant probes
    masked_density = trilinear_interpolation(volume, masked_probe_xyz_flat, voxel_size=voxel_size) # (n_valid_probes,)
    #masked_density = mm.density_at_sites_cart(masked_probe_xyz_flat_flex).as_numpy_array()
    
    # prepare an output array with zeros
    d_vals = np.zeros((n_shells, n_atoms, n_probes))
    
    # reshape interpolated values to (n_shells, n_atoms, n_probes) using the mask
    d_vals[keep_mask] = masked_density
    
    
    
    n_atoms = probe_xyz.shape[1]
    n_probes = probe_xyz.shape[2]
    M = volume
    maxD = min(M.mean()+M.std()*10,M.max())
    minD = max(M.mean()-M.std()*1,M.min())
    A = maxD-minD
    B = minD
    u = 0
    sigma = 0.6
    x = radii
    y = A * np.exp(-0.5*((x-u)/sigma)**2) + B 
    
    # stack the reference to shape (n_shells,n_atoms,n_probes)
    g_vals = np.repeat(y[:,None],n_probes,axis=1)
    x_repeat = np.repeat(x,n_probes)
    g_vals = np.expand_dims(g_vals,1)
    
    g_vals = np.tile(g_vals,(n_atoms,1))
    
    # Reshape to 2d for masked rowwise correlation calculation
    g_vals_2d = g_vals.transpose(1,0,2).reshape(g_vals.shape[1], -1)
    d_vals_2d = d_vals.transpose(1,0,2).reshape(d_vals.shape[1], -1)
    mask_2d = keep_mask.transpose(1,0,2).reshape(keep_mask.shape[1], -1)
    
    q = rowwise_corrcoef(g_vals_2d,d_vals_2d,mask=mask_2d)
    return q,probe_xyz,keep_mask,d_vals, g_vals


 