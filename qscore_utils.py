import numpy as np
import numpy.ma as ma

def sphere_points(ctr, rad, N):
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
  
def fisher(r):
    z = 0.5 * np.log((1 + r) / (1 - r))
    return z
  
def points_for_density(radii, density,min_value=1):
    return max(min_value,np.round(density * 4 * np.pi * radii**2).astype(int))

def stack_fill(arrs, dim=1, max_to_shape=None, fill=-1.0):
    max_shape = list(max(arr.shape for arr in arrs))
    if max_to_shape is not None:
        max_shape[np.argmax(max_shape)] = max_to_shape
    
    ret_arr = np.full((len(arrs),) + tuple(max_shape), fill)
    
    for i, arr in enumerate(arrs):
        # Slice of indices to target the correct dimensions
        indices = [slice(None)] * arr.ndim
        indices[dim] = slice(arr.shape[dim])
        if len(arr)>0:
            ret_arr[i][tuple(indices)] = arr
    
    return ret_arr

def standardize_selection_np(selection,model=None,n_atoms=None):
    # can get bool,int,or string. If int or string must pass model
    # Returns numpy bool selection

    if isinstance(selection,str):
        assert model is not None,"Must also pass model to use string selection"
        selection = model.selection(selection) # bool
    elif isinstance(selection,np.ndarray):
        if selection.dtype == int:
            assert model is not None or n_atoms is not None,"Must also pass model to use int selection"
            if model is not None:
              n_atoms = model.get_number_of_atoms()
            selection_bool = np.full(n_atoms,False)
            selection_bool[selection] = True
            selection = selection_bool
        elif selection.dtype == bool:
            pass
        else:
            assert False, "Unable to interpret selection"
    else:
        assert False, "Unable to interpret selection"

    return selection

def trilinear_interpolation(voxel_grid, coords, voxel_size=None, offset=None):
    assert voxel_size is not None, "Provide voxel size as an array or single value"
    
    # Apply offset if provided
    if offset is not None:
        coords = coords - offset

    # Transform coordinates to voxel grid index space
    index_coords = coords / voxel_size

    # Split the index_coords array into three arrays: x, y, and z
    x, y, z = index_coords.T

    # Truncate to integer values
    x0, y0, z0 = np.floor([x, y, z]).astype(int)
    x1, y1, z1 = np.ceil([x, y, z]).astype(int)

    # Ensure indices are within grid boundaries
    x0, y0, z0 = np.clip([x0, y0, z0], 0, voxel_grid.shape[0]-1)
    x1, y1, z1 = np.clip([x1, y1, z1], 0, voxel_grid.shape[0]-1)

    # Compute weights
    xd, yd, zd = [arr - arr.astype(int) for arr in [x, y, z]]

    # Interpolate along x
    c00 = voxel_grid[x0, y0, z0]*(1-xd) + voxel_grid[x1, y0, z0]*xd
    c01 = voxel_grid[x0, y0, z1]*(1-xd) + voxel_grid[x1, y0, z1]*xd
    c10 = voxel_grid[x0, y1, z0]*(1-xd) + voxel_grid[x1, y1, z0]*xd
    c11 = voxel_grid[x0, y1, z1]*(1-xd) + voxel_grid[x1, y1, z1]*xd

    # Interpolate along y
    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    # Interpolate along z
    c = c0*(1-zd) + c1*zd

    return c

def rowwise_corrcoef(A, B, mask=None):
    assert A.shape == B.shape, f"A and B must have the same shape, got: {A.shape} and {B.shape}"
    
    if mask is not None:
        assert mask.shape == A.shape, "mask must have the same shape as A and B"
        A = ma.masked_array(A, mask=np.logical_not(mask))
        B = ma.masked_array(B, mask=np.logical_not(mask))

    # Calculate means
    A_mean = ma.mean(A, axis=1, keepdims=True)
    B_mean = ma.mean(B, axis=1, keepdims=True)
    
    # Subtract means
    A_centered = A - A_mean
    B_centered = B - B_mean
    
    # Calculate sum of products
    sumprod = ma.sum(A_centered * B_centered, axis=1)
    
    # Calculate square roots of the sum of squares
    sqrt_sos_A = ma.sqrt(ma.sum(A_centered**2, axis=1))
    sqrt_sos_B = ma.sqrt(ma.sum(B_centered**2, axis=1))
    
    # Return correlation coefficients
    cc =  sumprod / (sqrt_sos_A * sqrt_sos_B)
    return cc.data