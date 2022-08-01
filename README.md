# Q-score
An implementation of the croe-EM validation metric Q-score using cctbx/numpy/scipy
Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7446556/

# Simple usage
```bash
python qscore.py --map=map.mrc --model=model.pdb
```
# Specify number of radial shells and probes-per-shell
```bash
python qscore.py --map=map.mrc --model=model.pdb --n_radial_shells=16 --n_probes_per_shell=8
```


# Only calculate for a selection
```bash
python qscore.py --map=map.mrc --model=model.pdb --selection="resname ALA"
```

# Write out plots of radial density profile
```bash
python qscore.py --map=map.mrc --model=model.pdb --selection="resname PTQ" --plots_file=plots.png
```
