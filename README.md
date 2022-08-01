# Q-score
An implementation of the croe-EM validation metric Q-score using cctbx/numpy/scipy

# Simple usage
```bash
python qscore.py --map=map.mrc --model=model.pdb
```
# Only calculate for a selection
```bash
python qscore.py --map=map.mrc --model=model.pdb --selection="resname ALA"
```

# Write out plots of radial density profile
```bash
python qscore.py --map=map.mrc --model=model.pdb --selection="resname PTQ" --plots_file=plots.png
```
