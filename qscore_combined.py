import math
from cctbx.array_family import flex
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
#from qscore_cctbx import query_atom_neighbors, query_ball_point_flex, cdist_flex
from qscore_utils import sphere_points

from tqdm.notebook import tqdm

def sphere_points_combined(ctr_np,ctr_flex,rad,N):

    h_np = -1.0 + (2.0 * np.arange(N) / float(N-1))[:, np.newaxis]
    h_flex = -1.0 + (2.0 * flex.double_range(N) / (N-1))
    #h_flex.reshape(flex.grid(N,1))
    assert np.all(np.isclose(h_np.flatten(),h_flex.as_numpy_array()))
    phis_flex = flex.acos(h_flex)
    phis_np = np.arccos(h_np)

    thetas_np = np.zeros_like(phis_np)
    thetas_flex = flex.double(len(phis_flex),0.0)
    a_flex = (3.6 / flex.sqrt(N * (1.0 - h_flex[1:-1]**2)))
    a_np = (3.6 / np.sqrt(N * (1.0 - h_np[1:-1]**2)))
    assert np.all(np.isclose(a_np.flatten(),a_flex.as_numpy_array()))

    thetas_np[1:-1, :] = a_np
    thetas_flex = thetas_flex.set_selected(flex.uint32_range(1,N-1),a_flex)
    assert np.all(np.isclose(thetas_np.flatten(),thetas_flex.as_numpy_array()))


    thetas_np = np.cumsum(thetas_np,axis=0)

    def cumsum_flex(arr):
        result = []
        running_sum = 0.0
        for i,x in enumerate(arr):
            running_sum += x
            result.append(running_sum)
        return flex.double(result)

    thetas_flex = cumsum_flex(thetas_flex)

    assert np.all(np.isclose(thetas_np.flatten(),thetas_flex.as_numpy_array()))


    x_np = np.sin(phis_np) * np.cos(thetas_np)
    y_np = np.sin(phis_np) * np.sin(thetas_np)
    z_np = np.cos(phis_np)

    x_flex = flex.sin(phis_flex) * flex.cos(thetas_flex)
    y_flex = flex.sin(phis_flex) * flex.sin(thetas_flex)
    z_flex = flex.cos(phis_flex)


    # Stack x, y, z to form points and multiply by rad
    points_np = rad * np.stack([x_np, y_np, z_np], axis=-1)

    points_flex = rad * flex.vec3_double(x_flex,y_flex,z_flex)
    
    assert np.all(np.isclose(points_np.flatten(),points_flex.as_numpy_array().flatten()))


    # put all together

    # numpy
    points_np = points_np.reshape(1, N, 3)
    # add to ctr
    points_np = ctr_np[:, np.newaxis, :] + points_np


    # flex
    def broadcast_add(ctr_flex, points_flex):
        N = points_flex.size()
        M = ctr_flex.size()
        # Preallocate an array of shape (M*N, 3)
        result = flex.vec3_double(M*N)

        for i in range(M):
            for j in range(N):
                flat_index = i * N + j
                new_point = tuple(ctr_flex[i:i+1] + points_flex[j:j+1])[0]
                result[flat_index] = new_point


        result = result.as_1d().as_double()
        result.reshape(flex.grid(len(ctr_flex),len(points_flex),3))
        return result
    
    
    points_flex = broadcast_add(ctr_flex,points_flex)

    #assert np.all(np.isclose(points_np.ravel(),points_flex.as_double().as_numpy_array()))
    return points_np, points_flex

def radial_shell_worker_combined(args):
    i,model,n_probes,n_probes_target,radius_shell,rtol, selection = args
    
    atoms_xyz_flex = model.get_sites_cart()
    atoms_xyz = atoms_xyz_flex.as_numpy_array()
    args = 0,model, 16,8,2,0.9 
    i=0
    n_probes = 16
    n_probes_target=8
    radius_shell=2.0
    rtol = 0.9
    numPts = n_probes_target
    RAD = radius_shell
    outRAD = rtol

    #query neighbors
    inds_flex,dists_flex = query_atom_neighbors(model,radius=5)
    kdtree_atoms= KDTree(atoms_xyz)
    inds = kdtree_atoms.query_ball_point(atoms_xyz,5)
    # test agree
    for i in range(len(atoms_xyz)):
        inds_flex_i = list(inds_flex[i])
        inds_i = inds[i]
        assert set(inds_flex_i)==set(inds_i), "Mismatch in neighbor search"

    # manage selection
    if selection is None:
        selection = flex.bool(len(atoms_xyz),True)
        
        
    selection_flex = flex.bool(selection)
    selection_np = selection_flex.as_numpy_array()
    n_atoms = np.sum(selection_np)
    
    atoms_xyz_sel_np = atoms_xyz[selection_np]
    atoms_xyz_sel_flex = selection_flex.select(selection_flex)
    n_atoms = selection.count(True)
    
    # build out data structures

    # numpy
    probe_xyz_np = np.full((n_atoms,n_probes_target,3),-1.0)

    # mask is later

    #cctbx
    probe_xyz_flex = flex.double(n_atoms*n_probes_target*3,-1.0) # init flat xyz array
    probe_xyz_flex.reshape(flex.grid(n_atoms,n_probes_target,3))

    keep_mask_flex = flex.bool(n_atoms*n_probes_target,False)
    keep_mask_flex.reshape(flex.grid(n_atoms,n_probes_target))


    # loop once per atom
    all_pts_np = []
    all_pts_flex = []
    for atom_i,_ in enumerate(range(n_atoms)):
        coord_np = atoms_xyz_sel_np[atom_i:atom_i+1]
        coord_flex = atoms_xyz_sel_flex[atom_i:atom_i+1]

        pts_np = []
        pts_flex = []
        
        # try to get at least [numPts] points at [RAD] distance
        # from the atom, that are not closer to other atoms
        for i in range (0, 50) :
            # if we find the necessary number of probes in the first iteration, then i will never go to 1
            # points on a sphere at radius RAD...
            n_pts_to_grab = numPts+i*2 # progressively more points are grabbed  with each failed iter
            #print("n_to_grab:",n_pts_to_grab)
            outPts_np = sphere_points(coord_np,RAD,n_pts_to_grab)[0] # get the points
            outPts_flex = flex.double(outPts_np)
            #outPts_flex_double = outPts_flex.as_double()
            outPts_flex.reshape(flex.grid(n_pts_to_grab,3))

            # make empty lists
            at_pts_i_np = 0 
            at_pts_np =  [None]*len(outPts_flex)

            at_pts_i_flex = 0
            at_pts_flex = [None]*len(outPts_flex)

            # find atom clashes
            for pt_i in range(n_pts_to_grab): # identify which ones to keep, progressively grow pts list
                pt_np = outPts_np[pt_i:pt_i+1]

                pt_flex = outPts_flex.select(flex.uint32([pt_i*3+0,
                                                      pt_i*3+1,
                                                      pt_i*3+2,
                                                     ]))
                pt_flex.reshape(flex.grid(1,3))


                nbrs = inds[atom_i]
                nbrs_xyz_np = atoms_xyz[nbrs]
                nbrs_xyz_flex = atoms_xyz_flex.select(flex.uint32(nbrs)).as_double()
                nbrs_xyz_flex.reshape(flex.grid(len(nbrs),3))

                # test select
                assert np.all(nbrs_xyz_np==nbrs_xyz_flex.as_numpy_array())

                # test distance
                d_flex = cdist_flex(outPts_flex,pt_flex)
                d_np = cdist(outPts_np,pt_np)
                assert np.all(np.isclose(d_np,d_flex.as_numpy_array()))



                # test brute force count
                #count_np = query_ball_point(atoms_xyz,pt_np[0],RAD*outRAD)
                count_flex = query_ball_point_flex(nbrs_xyz_flex,pt_flex,RAD*outRAD)

                # test kdtree count
                count_np = kdtree_atoms.query_ball_point(pt_np[None,:],RAD*outRAD,return_length=True).flatten()[0]

                assert count_np==count_flex


                # if no clashes, add to list
                # np
                if count_np == 0 :
                    at_pts_np[at_pts_i_np] = pt_np
                    at_pts_i_np += 1

                # cctbx
                if count_flex == 0 :
                    at_pts_flex[at_pts_i_flex] = pt_flex
                    at_pts_i_flex += 1

            if at_pts_i_np >= numPts : # if we have enough points, take all the "good" points from this iter
                pts_np.extend ( at_pts_np[0:at_pts_i_np] )


                if at_pts_i_flex >= numPts : # if we have enough points, take all the "good" points from this iter
                    pts_flex.extend ( at_pts_flex[0:at_pts_i_flex] )
                    break
                else:
                    assert False, "numpy flex disagreement"

        #add points to array for each atom.

        # numpy
        if len(pts_np)>0:
            probe_xyz_np[atom_i] = np.vstack(pts_np)[:n_probes_target]
        
        # cctbx
        if len(pts_flex)>0:
            for j,pt in enumerate(pts_flex):
                if j<n_probes_target:
                    x,y,z = pt
                    probe_xyz_flex[atom_i,j,0] = pt[0]
                    probe_xyz_flex[atom_i,j,1] = pt[1]
                    probe_xyz_flex[atom_i,j,2] = pt[2]
                    keep_mask_flex[atom_i,j] = True



    # the numpy keep mask is calculated different
    keep_sel = probe_xyz_np != -1.
    keep_sel = np.mean(keep_sel, axis=-1, keepdims=True)  
    keep_mask_np = np.squeeze(keep_sel, axis=-1).astype(bool)

    # tests
    assert np.all(np.isclose(probe_xyz_np,probe_xyz_flex.as_numpy_array()))
    assert np.all(keep_mask_np==keep_mask_flex.as_numpy_array())

    return probe_xyz_np, probe_xyz_flex, keep_mask_np, keep_mask_flex


def radial_shell_combined_mp(model,n_probes=64,n_probes_target=8,radii=None,rtol=0.9,num_processes=cpu_count(),selection=None):

    # Create argument tuples for each chunk
    args = [(i,model,n_probes,n_probes_target,radius_shell,rtol,selection) for i,radius_shell in enumerate(radii)]
    
    # Create a pool of worker processes
    if num_processes >1:
        with Pool(num_processes) as p:
            # Use the pool to run the trilinear_interpolation_worker function in parallel
            results = p.map(radial_shell_worker_combined, args)
    else:
        results = []
        for arg in tqdm(args):
            result = radial_shell_worker_combined(arg)
            results.append(result)

    
    # stack numpy
    probe_xyz_all = [result[0] for result in results]
    keep_mask_all = [result[2] for result in results]
    

    n_shells = len(radii)
    n_atoms = probe_xyz_all[0].shape[0]
    out_shape = (n_shells,n_atoms,n_probes,3 )
    out_size = np.prod(out_shape)
    shell_size = np.prod(out_shape[1:])
    out_probes = np.full((n_shells,n_atoms,n_probes_target,3),-1.0)
    out_mask = np.full((n_shells,n_atoms,n_probes_target),False)
    
    for i,p in enumerate(probe_xyz_all):
        out_probes[i,:,:n_probes_target,:] =p[:,:n_probes_target]

    
    for i,k in enumerate(keep_mask_all):
        start = i*shell_size
        stop = start+shell_size
        out_mask[i] = k[:,:n_probes_target]
    probe_xyz_np = out_probes
    keep_mask_np = out_mask
    
    # stack flex
    probe_xyz_all = [result[1] for result in results]
    keep_mask_all = [result[3] for result in results]
    
    
    n_shells = len(radii)
    n_atoms = probe_xyz_all[0].focus()[0]
    out_shape = (n_shells,n_atoms,n_probes_target,3 )
    out_size = math.prod(out_shape)
    shell_size = math.prod(out_shape[1:])
    out_probes = flex.double(out_size,-1.0)
    out_mask = flex.bool(n_atoms*n_shells*n_probes_target,False)
    for i,p in enumerate(probe_xyz_all):
        start = i*shell_size
        stop = start+shell_size
        out_probes = out_probes.set_selected(flex.uint32_range(start,stop),p.as_1d())
    out_probes.reshape(flex.grid(*out_shape))

    
    for i,k in enumerate(keep_mask_all):
        start = i*(n_atoms*n_probes_target)
        stop = start+(n_atoms*n_probes_target)
        out_mask = out_mask.set_selected(flex.uint32_range(start,stop),k.as_1d())
    out_mask.reshape(flex.grid(n_shells,n_atoms,n_probes_target))
    
    probe_xyz_flex = out_probes
    keep_mask_flex = out_mask
    
    return probe_xyz_np, probe_xyz_flex, keep_mask_np, keep_mask_flex




def Qscore1_combined(mmm,
                    radial_shell_func,
                    voxel_size=1.0,
                    n_probes=8,
                    min_probes=1,
                    radii=np.arange(0.1,2.1,0.1),
                    rtol=0.9,
                    selection=None,
                    num_processes=cpu_count()):
    
    radii = [r if r != 0 else 1e-9 for r in radii]
    
    model = mmm.model()
    mm = mmm.map_manager()
    M = mm.map_data()
    
    # (probe_xyz, 
    #  probe_xyz_cctbx, 
    #  keep_mask, 
    #  keep_mask_cctbx) = radial_shell_combined_mp(model,
    #                                               n_probes=n_probes,
    #                                               radii=radii,
    #                                               rtol = rtol,
    #                                               selection=selection,
    #                                               num_processes=num_processes)
    
    (probe_xyz, 
     probe_xyz_cctbx, 
     keep_mask, 
     keep_mask_cctbx) = radial_shell_combined_mp(model,
                                                  n_probes=n_probes,
                                                  radii=radii,
                                                  rtol = rtol,
                                                  selection=selection,
                                                  num_processes=num_processes)
    
    
    
    # debug
    do_np = True
    do_flex = True
    do_test = True

    # to test must do both
    if do_test:
        do_np = True
        do_flex = True
    # PROCEED



    # flatten numpy arrays
    if do_np:
        n_shells,n_atoms,n_probes,_ = probe_xyz.shape
        probe_xyz_flat = probe_xyz.reshape((n_atoms*n_shells*n_probes,3))
        keep_mask_flat = keep_mask.reshape(-1) # (n_shells*n_atoms*n_probes,)


    # init cctbx arrays

    if do_flex:
        n_shells,n_atoms,n_probes,_ = probe_xyz_flex.focus()
        probe_xyz_cctbx = flex.double(probe_xyz)
        keep_mask_cctbx = flex.bool(keep_mask)




    # APPLY MASK BEFORE INTERPOLATION

    # numpy
    if do_np:
        masked_probe_xyz_flat = probe_xyz_flat[keep_mask_flat]

    # cctbx
    if do_flex:
        keep_mask_cctbx_fullflat = []

        for val in keep_mask_cctbx:
            for _ in range(3):  # since A has an additional dimension of size 3
                keep_mask_cctbx_fullflat.append(val)

        mask = flex.bool(keep_mask_cctbx_fullflat)
        #indices = flex.int([i for i in range(1, keep_mask_cctbx.size() + 1) for _ in range(3)])
        sel = probe_xyz_cctbx.select(mask)
        #sel_indices = indices.select(mask)
        masked_probe_xyz_flat_cctbx = flex.vec3_double(sel)

        
        # INTERPOLATE

    # numpy
    if do_np:
        M = mm.map_data()
        volume = M.as_numpy_array()
        voxel_size = np.array(mm.pixel_sizes())
        masked_density = trilinear_interpolation(volume, masked_probe_xyz_flat, voxel_size=voxel_size)

    # cctbx
    if do_flex:
        masked_density_cctbx = mm.density_at_sites_cart(masked_probe_xyz_flat_cctbx)

        # test equivalent
        if do_test:
            assert np.all(np.isclose(masked_density,masked_density_cctbx.as_numpy_array()))


    # reshape interpolated values to (n_shells,n_atoms, n_probes)

    # numpy
    if do_np:
        d_vals = np.zeros((n_shells, n_atoms, n_probes))
        d_vals[keep_mask] = masked_density

    # cctbx
    if do_flex:
        keep_mask_cctbx.reshape(flex.grid(n_shells*n_atoms*n_probes))
        d_vals_cctbx = flex.double(keep_mask_cctbx.size(),0.0)
        d_vals_cctbx = d_vals_cctbx.set_selected(keep_mask_cctbx,masked_density_cctbx)
        d_vals_cctbx.reshape(flex.grid(n_shells,n_atoms,n_probes))

        if do_test:
            # test
            assert np.all(np.isclose(d_vals,d_vals_cctbx.as_numpy_array()))


    # reshape to (M,N*L) for rowwise correlation

    # numpy
    if do_np:
        d_vals_2d = d_vals.transpose(1,0,2).reshape(d_vals.shape[1], -1)

    # cctbx
    if do_flex:
        def custom_reshape_indices(flex_array):
            N,M,L = flex_array.focus()
            result = flex.double(flex.grid(M, N * L))

            for i in range(N):
                for j in range(M):
                    for k in range(L):
                        # Calculate the original flat index
                        old_index = i * M * L + j * L + k
                        # Calculate the new flat index after transpose and reshape
                        new_index = j * N * L + i * L + k
                        result[new_index] = flex_array[old_index]

            return result

        d_vals_2d_cctbx = custom_reshape_indices(d_vals_cctbx)


    # test
    if do_test:
        assert np.all(np.isclose(d_vals_2d,d_vals_2d_cctbx.as_numpy_array()))


    # create the reference data

    # numpy
    if do_np:
        M = mm.map_data().as_numpy_array()
        maxD = min(M.mean()+M.std()*10,M.max())
        minD = max(M.mean()-M.std()*1,M.min())
        A = maxD-minD
        B = minD
        u = 0
        sigma = 0.6
        x = np.array(radii)
        y = A * np.exp(-0.5*((x-u)/sigma)**2) + B 




    #cctbx
    if do_flex:
        M = mm.map_data()
        maxD_cctbx = min(flex.mean(M)+M.standard_deviation_of_the_sample()*10,flex.max(M))
        minD_cctbx = max(flex.mean(M)-M.standard_deviation_of_the_sample()*1,flex.min(M))
        A_cctbx = maxD_cctbx-minD_cctbx
        B_cctbx = minD_cctbx
        u = 0
        sigma = 0.6
        x = flex.double(radii)
        y_cctbx = A_cctbx * flex.exp(-0.5*((flex.double(x)-u)/sigma)**2) + B_cctbx




        # test
        if do_test:
            assert np.all(np.isclose(np.array(y_cctbx),y))


    # Stack and reshape data for correlation calc

    # numpy
    if do_np:
        # stack the reference to shape (n_shells,n_atoms,n_probes)
        g_vals = np.repeat(y[:,None],n_probes,axis=1)
        g_vals = np.expand_dims(g_vals,1)
        g_vals = np.tile(g_vals,(n_atoms,1))


        # reshape
        g_vals_2d = g_vals.transpose(1,0,2).reshape(g_vals.shape[1], -1)
        d_vals_2d = d_vals.transpose(1,0,2).reshape(d_vals.shape[1], -1)
        mask_2d = keep_mask.transpose(1,0,2).reshape(keep_mask.shape[1], -1)



    # cctbx
    if do_flex:
        # 1. Repeat y for n_probes (equivalent to np.repeat)
        g_vals_cctbx = [[val] * n_probes for val in y_cctbx]

        # 2. Add a new dimension (equivalent to np.expand_dims)
        g_vals_expanded = [[item] for item in g_vals_cctbx]

        # 3. Tile for each atom (equivalent to np.tile)
        g_vals_tiled = []
        for item in g_vals_expanded:
            g_vals_tiled.append(item * n_atoms)


        g_vals_cctbx = flex.double(np.array(g_vals_tiled) )


        # test 
        if do_test:
            assert np.all(np.isclose(g_vals_cctbx.as_numpy_array(),g_vals))


    # # CALCULATE Q

    # # numpy
    if do_np:
        q = rowwise_corrcoef(g_vals_2d,d_vals_2d,mask=mask_2d)



    # cctbx
    if do_flex:
        d_vals_cctbx = d_vals_cctbx.as_1d()
        g_vals_cctbx = g_vals_cctbx.as_1d()
        keep_mask_cctbx_double = keep_mask_cctbx.as_1d().as_double()
        q_cctbx = []
        for atomi in range(n_atoms):

            #inds = nd_to_1d_indices((None,atomi,None),(n_shells,n_atoms,n_probes))
            inds = optimized_nd_to_1d_indices(atomi,(n_shells,n_atoms,n_probes))
            inds = flex.uint32(inds)
            d_row = d_vals_cctbx.select(inds)

            if do_test:
                assert np.all(np.isclose(d_row.as_numpy_array(),d_vals_2d[atomi]))

            g_row = g_vals_cctbx.select(inds)
            if do_test:
                assert np.all(np.isclose(g_row.as_numpy_array(),g_vals_2d[atomi]))

            mask = keep_mask_cctbx.select(inds)
            if do_test:
                assert np.all(np.isclose(mask.as_numpy_array(),mask_2d[atomi]))

            d = d_row.select(mask)
            g = g_row.select(mask)
            qval = flex.linear_correlation(d,g).coefficient()
            q_cctbx.append(qval)


        q_cctbx = flex.double(q_cctbx)
        if do_test:
            assert np.all(np.isclose(q,np.array(q_cctbx)))

    return q, q_cctbx