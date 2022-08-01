import sys
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from iotbx.data_manager import DataManager
from cctbx.array_family import flex

__doc__ = """
This file aims to implement Q-score from this paper:  based on the paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7446556/

TODO: Write Q somewhere... to a new mmcif field?
TODO: Is the correlation calculation the same? I seem to get higher values than the paper...
TODO: Is using a fixed number of probes at each shell the best approach? At 2 angstroms 8 probes is very sparse sampling of the density...
TODO: This could be sped up by not calculating atom-probe proximity for probes at larger radius shells if the corresponding probe was
      already rejected at a lower radius shell.
"""



def radial_density_average(map_model_manager,
                   r=1.0,
                   n=8,
                   rtol=0.05,
                   selection="all"):
    """
    Get a radial average of map density values for probe points around atoms.
    Does not include values for probe values which clash with other atoms
    
    Arguments:
        map_model_manager: 
            cctbx map model manager
        r: 
            radius at which to sum the shell
        n:
            number of probes on the sphere
        rtol:
            A small positive value is necessary to add to the radius
                when checking for neighbor atom to probe clashes
        selection:
            Phenix string selection, q will only be calculated for these atoms
            
    Returns: 
        density_average:
            Average map value at radius r from each atom (n_atoms)
        probe_xyz:
            cartesion coords of all non-rejected probe values shape: (n_atoms * n_probes)-n_rejected
        atoms:
            list of each cctbx atom for which Q was calculated
    """
    
    
    
    # Extract objects
    mmm = map_model_manager
    model = mmm.model()
    sel = model.selection(selection).as_numpy_array()
    assert np.sum(sel)>0, "Selection returned zero atoms"

    # make sklearn kdtree for fast distance lookup
    xyz = model.get_sites_cart().as_numpy_array()
    n_atoms = xyz.shape[0]
    n_atoms_sel = xyz[sel].shape[0]
    kdtree_atoms = KDTree(xyz)

    # get a set of probes at a radius around zero
    unit_probes = radial_points(samples=n,rad=r)
    n_probes = unit_probes.shape[0]

    # broadcast to each atom and translate to atom center
    probe_xyz = (xyz[sel][:,np.newaxis,:]+unit_probes[np.newaxis,:]) # (n_atoms,n_probes,3)

    # query kdtree to find probe-atom interactions (the slowest part by far)
    counts = kdtree_atoms.query_ball_point(probe_xyz,r+rtol,return_length=True) #  (n_atoms,n_probes) (a count value for each probe)
    # each value in counts is the number of atoms within radius+tol of each probe

    # Only want to select probes with a single atom neighbor
    keep_sel = counts==1

    # Use cctbx to interpolate map values
    keep_probe_cart =flex.vec3_double(probe_xyz[keep_sel]) # cartesion coords of each kept probe 
    probe_density_flat = mmm.map_manager().density_at_sites_cart(keep_probe_cart).as_numpy_array() # map value at each probe point 


    # create empty density array for all probes, fill with zeros
    probe_density = np.full((n_atoms_sel,n_probes),0.0)
    probe_density[keep_sel] = probe_density_flat # set kept density values


    # Average map value for each atom using each non-rejected probe value

    # use 1 or 0 weight to mask out rejected probes
    weights = np.full((n_atoms_sel,n_probes),1e-12) # small non-zero weight to avoid weight sum = 0
    weights[keep_sel] = 1.0 # set kept probes to weight ==1
    density_average = np.average(probe_density,weights=weights,axis=1)

    # select probes to return
    probe_xyz = probe_xyz[keep_sel]

    # select atoms to return (for convenience)
    atoms = mmm.model().get_atoms()
    atoms = [atom for i,atom in zip(sel,atoms) if i]
    
    return density_average,probe_xyz,atoms
        
    
def radial_points(samples=1000,rad=1.0):
    """
    Sample points evenly around the surface of a sphere 
    with radius r centered at (0,0,0)
    
    This fibonacci method should be more uniform than sampling
    on Euler angles, and doesn't require introducing quaternions
    
    Reference:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
    phi = np.pi * (3. - np.sqrt(5.)) 
    i = np.arange(samples)
    y = 1. - (i / float(samples - 1.)) * 2.

    radius = np.sqrt(1. - y * y) 
    theta = phi * i  
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    points = np.vstack([x,y,z]).T
    points*=rad
    assert np.all(np.isclose(np.linalg.norm(points,axis=1),rad)), "Points are not equidistance from center by specified radius"
    return points



def points_to_bild(points,outfile="debug.bild",radius=0.1,color="purple"):
    """
    Write a .bild file to vizualize radial points in Chimera.
    
    Args:
        points: np array of shape (n_points,3)
        outfile: File path to write out
        radius: radius to use when depicting the points
        color: color of points. One of the Chimera color keywords
    """
    bild_lines = []
    for p in np.vstack(points):
        s = ".color "+color+"\n.sphere "+" ".join(str(v) for v in p)+" %s"%str(radius)+"\n"
        bild_lines.append(s)
    with open(outfile,"w") as fh:
        fh.writelines(bild_lines)
        
        
        
def plot_profiles(q,y,radial_profiles,atoms,normalize=True,show_atom_identifier=True,max_plots=200,reverse_sort=False,filename=None):
    """
    Plot radial density to debug Q calculations
    """
    # set atom convenvience identifiers
    for atom in atoms:
        if not hasattr(atom.__class__,"identifier_dict"):
            setattr(atom.__class__,"identifier_dict",property(lambda self: {
                "asym_id":self.parent().parent().parent().id.strip(),
                "seq_id":self.parent().parent().resseq_as_int(),
                "comp_id":self.parent().resname.strip(),
                "atom_id":self.name.strip(),
                "alt_id":self.parent().altloc.strip()}))
        
        
    sort = np.flip(np.argsort(q))
    if reverse_sort:
        sort = np.flip(sort)
    q = q[sort]
    radial_profiles = radial_profiles[sort]
    
    if len(radial_profiles)>max_plots:
        #assert False, "Number of radial profiles greater than specified max. Large numbers of plots can be very slow"
        radial_profiles = radial_profiles[:max_plots]
        q = q[:max_plots]
    
    if normalize:
        y = (y-y.min()) / (y.max()-y.min())
        radial_profiles = (radial_profiles-radial_profiles.min(axis=1)[:,np.newaxis]) / (radial_profiles.max(axis=1) - radial_profiles.min(axis=1))[:,np.newaxis]

    atoms = [atoms[i] for i in sort]
    rows = int(len(radial_profiles)/6)+1
    fig, axs = plt.subplots(rows,6,figsize=(16,max(2,int(0.5*len(radial_profiles)))))
    axs = axs.flatten()
    for i in range(len(radial_profiles)):
        ax = axs[i]
        x = radial_profiles[i]
        ax.plot(rs,x,marker="o",color="purple")
        atom = atoms[i]
        title = "Q="+str(round(q[i],4))
        ax.set_title(title)
        #ax.set_aspect("equal")
        ax.plot(rs,y,color="green")
        if show_atom_identifier:
            ident = "atom_idx="+str(sort[i])+"\n"+"\n".join([key+"="+str(value) for key,value in atom.identifier_dict.items()])
            ax.text(1, (y.min()+y.max())/3, ident, fontsize = 8)
    
    
    fig.tight_layout()
    if filename is not None:
        plt.savefig(str(filename))
        
        
        
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Calculate 'Q-score', a map-model validation metric.")
    argparser.add_argument('--model', type=str, help="Path to a model file (pdb/mmcif)")
    argparser.add_argument('--map', type=str, help="Path to a map file (mrc/ccp4/map)")
    argparser.add_argument('--n_radial_shells', type=int,default=16, help="Number of radial shells between 0 and 2 angstroms")
    argparser.add_argument('--n_probes_per_shell', type=int,default=8, help="Number of radial probes per shell")
    argparser.add_argument('--selection', type=str,default="all", help="CCTBX/Phenix selection string. Only calculate Q for these atoms.")
    argparser.add_argument('--plots_file', type=str,default="", help="Plot the radial profiles for each atom, sorted by Q")
    argparser.add_argument('--plots_normalize', type=bool, help="Normalize the reference profile and the atom profile 0-1")
    argparser.add_argument('--quiet', type=bool,default=False, help="Suppress all output")
    args = argparser.parse_args()
    
    if [args.model,args.map].count(None)>0:
        argparser.print_help()
        sys.exit()
        
        
    dm = DataManager()
    dm.process_model_file(args.model)
    dm.process_real_map_file(args.map)
    mmm = dm.get_map_model_manager()
    assert flex.sum(mmm.map_manager().density_at_sites_cart(mmm.model().get_sites_cart()))>0,(
    "Negative or zero density around atoms. Map/model likely not aligned")
    
    # calculate radial profiles v

    radial_profiles = [] # radial profile array for each radius r
    atoms_all = [] # atoms for each radius r (redundant)
    probes_xyz = [] # probe coordinates for each radius r

    n_r = args.n_radial_shells
    rs = np.linspace(0,2,n_r)
    for r in tqdm.tqdm(rs,disable=args.quiet):
        profile, probe_xyz, atoms = radial_density_average(mmm,r=r,n=args.n_probes_per_shell,selection=args.selection)
        atoms_all.append(atoms)
        radial_profiles.append(profile)
        probes_xyz.append(probe_xyz)

    radial_profiles = np.array(radial_profiles).T # to (n_atoms_sel,n_probes)
    atoms = atoms_all[0]
    
    # calculate q
    mm = mmm.map_manager()
    map_values = mm.map_data().as_numpy_array()
    Mavg = map_values.mean()
    Msigma = map_values.std()
    A = Mavg + 10*Msigma
    B = Mavg - 1.0*Msigma
    u = 0
    sigma = 0.6
    x = rs
    y = A * np.exp(-0.5*((x-u)/sigma)**2) + B # u
    y_flex = flex.double(list(y))
    
    # q for each radial profile
    q = np.array([flex.linear_correlation(flex.double(list(x)),y_flex).coefficient() for x in radial_profiles])
    
    # print for now instead of writing
    # set atom convenvience identifiers
    if not args.quiet:
        for i,atom in enumerate(atoms):
            if not hasattr(atom.__class__,"identifier_dict"):
                setattr(atom.__class__,"identifier_dict",property(lambda self: {
                    "asym_id":self.parent().parent().parent().id.strip(),
                    "seq_id":self.parent().parent().resseq_as_int(),
                    "comp_id":self.parent().resname.strip(),
                    "atom_id":self.name.strip(),
                    "alt_id":self.parent().altloc.strip()}))
            print("Atom:"+str(i)+","+", ".join([key+":"+str(value) for key,value in atom.identifier_dict.items()])+", Q:"+str(round(q[i],4)))
    
    # write bild (for now, the middle radial shell)
    i = int(len(rs)/2)+1
    r = rs[i]
    points = probes_xyz[i]
    points_to_bild(points,"q_probes.bild")

    # plot
    if len(args.plots_file)>0:
        
        plot_profiles(q,
                      y,
                      radial_profiles,
                      atoms,
                      normalize=args.plots_normalize,
                      show_atom_identifier=True,
                      reverse_sort=False,
                      filename=args.plots_file)
    
    
