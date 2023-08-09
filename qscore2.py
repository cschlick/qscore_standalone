import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Pool,cpu_count
from qscore_utils import (
          sphere_points,
          stack_fill,
          trilinear_interpolation,
          rowwise_corrcoef,
          )



def radial_shell_worker(args):
    i,atoms_xyz,n_probes,radius_shell,rtol= args

    sel = np.full(atoms_xyz.shape[0],True)
    
    kdtree_atoms = KDTree(atoms_xyz)

    probe_xyz_r = sphere_points(atoms_xyz,radius_shell,n_probes)
    
    # query kdtree to find probe-atom interactions (the slowest part by far)
    counts = kdtree_atoms.query_ball_point(probe_xyz_r,radius_shell*rtol,return_length=True) #  (n_atoms,n_probes) (a count value for each probe)
    # each value in counts is the number of atoms within radius+tol of each probe
    
    # Only want to select probes with a single atom neighbor
    keep_sel = counts==0

    return (probe_xyz_r, keep_sel)

def radial_shell_mp(atoms_xyz,n_shells=21,n_probes=64,radii=None,rtol =1.1,num_processes=4):



    # Create argument tuples for each chunk
    args = [(i,atoms_xyz,n_probes,radius_shell,rtol) for i,radius_shell in enumerate(radii)]

    # Create a pool of worker processes
    with Pool(num_processes) as p:
        # Use the pool to run the trilinear_interpolation_worker function in parallel
        results = p.map(radial_shell_worker, args)

    # stackthe results from each process
    probe_xyz = np.stack([result[0] for result in results])
    keep_mask = np.stack([result[1] for result in results])
    return probe_xyz,keep_mask
def balance_bool_rows(a, target):
    """
    a: A 2D boolean array.
    target: The target number of True values in each row.
    """
    # This operation will set some True values to False in each row, so that
    # the number of True values is approximately the target.
    for i in range(a.shape[0]):
        true_indices = np.where(a[i])[0]
        num_true = true_indices.size
        if num_true > target:
            # Randomly select excess True values and set them to False
            false_indices = np.random.choice(true_indices, size=num_true-target, replace=False)
            a[i, false_indices] = False
    return a

def Qscore2(volume,
            atoms_xyz,
            mask_clash=True,
            voxel_size=1.0,
            n_shells=21,
            n_probes=8,
            radius=2.0,
            min_probes=1,
            radii=None,
            rtol=0.9,
            ignore_min_probes=False,
            selection_bool=None,
            num_processes=cpu_count()):
    
    # handle selection at the very beginning
    if selection_bool is None:
      selection_bool = np.full(atoms_xyz.shape[0],True)
      
    atoms_xyz = atoms_xyz[selection_bool]
    if radii is None:
        rads = np.linspace(0,radius,n_shells)
    else:
        rads = radii

    probe_xyz,keep_mask = radial_shell_mp(atoms_xyz,
                                          rtol=rtol,
                                          n_shells=n_shells,
                                          radii=rads,
                                          n_probes=n_probes,
                                          num_processes=num_processes)


    n_shells,n_atoms,n_probes,_ = probe_xyz.shape

    # find atom/shell combinations where no probes were found. make sure those shells are distant from atom
    
    # keep_mask is a boolean array(n_shells,n_atoms,n_probes)
    keep_mask_debug =keep_mask.reshape(-1,keep_mask.shape[2]) # (n_shells*n_atoms,n_probes)
    is_blank = np.all(~keep_mask_debug,axis=1)
    n_blanks = is_blank.sum()
    is_blank_reshaped = is_blank.reshape(keep_mask.shape[0], keep_mask.shape[1])
    
    # find the n_shells dim 0 value in keep_mask for each true value in is_blank
    shell_index_blank, _  = np.where(is_blank_reshaped)
    shell_index_blank = shell_index_blank
    #assert rads[shell_index_blank.min()]>1.4 # make sure distant from atom
    if n_blanks>0:
      print("Closest blank:",rads[shell_index_blank.min()])

    #interpolate density 
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
    x = rads
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

    # balance
    #mask_2d = balance_bool_rows(mask_2d,8)
              
    q = rowwise_corrcoef(g_vals_2d,d_vals_2d,mask=mask_2d)
    return q,probe_xyz,keep_mask,d_vals,g_vals