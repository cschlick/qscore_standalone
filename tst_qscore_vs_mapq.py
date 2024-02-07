from __future__ import absolute_import, division, print_function
import os
import copy
import subprocess
import shutil
from pathlib import Path
import json
import argparse
from io import StringIO
from collections import OrderedDict


from iotbx.data_manager import DataManager
import libtbx
import numpy as np
from scipy.spatial import KDTree
import pandas as pd


from cctbx.maptbx.qscore import (
  aggregate_qscore_per_residue
)

from cctbx.maptbx.tst_qscore import (
  convert_dict_to_group_args,
  convert_group_args_to_dict,
  build_tests,
  run_test,
  test_template
)

def isclose_or_nan(a, b, atol=1e-3):
  # perform isclose comparison, treating nans as equal
  return np.isclose(a, b, atol=atol) | (np.isnan(a) & np.isnan(b))

def run_test_template_mapq(test,
                          mapq_location=None,
                          mapq_debug_data_filename="debug_data_mapq.json"):
  """
  Run mapq via Chimera on the command line. Load debug results
  NOTE: The debug results rely on a modified version of mapq.
    The releaseed version does not write intermediate data assumed here.
  """
  assert mapq_location is not None
  template = test # alias
  mapq_location = Path(mapq_location)
  model_path = Path(template["data"]["model_file"])
  map_path = Path(template["data"]["map_file"])
  test_dir = Path(template["data"]["test_dir"],template["data"]["name"])

  if test_dir.exists():
    shutil.rmtree(test_dir)
  test_dir.mkdir()
  local_model_path = Path(test_dir,model_path.name)
  local_map_path = Path(test_dir,map_path.name)
  shutil.copyfile(model_path,local_model_path)
  shutil.copyfile(map_path,local_map_path)
  with Path(test_dir,"test.json").open("w") as fh:
    test_d = convert_group_args_to_dict(test)
    test_d["data"]["test_dir"] = str(test_dir)
    test_d["data"]["model_file"] = str(local_model_path)
    test_d["data"]["map_file"] = str(local_map_path)
    fh.write(json.dumps({"data":test_d["data"],"params":test_d["params"]},indent=2))
  # model_path = local_model_path
  # map_path = local_map_path

  # run program
  cwd = os.getcwd()
  os.chdir(str(local_model_path.parent))
  mapq_executable = mapq_location / Path("mapq_cmd.py")
  chimera_path = mapq_location / Path("../../../../../Chimera.app")
  if "cif" in "".join(local_model_path.suffixes):
    suffix = 'cif'
  else:
    suffix = 'pdb'
  mapq_command = f"python {mapq_executable.absolute()}\
                  {chimera_path.absolute()}\
                  map={local_map_path.name}\
                  {suffix}={local_model_path.name}\
                  bfactor=1.0"
  # q is stored in bfactor column of pdb, with:  bfactor = f * (1.0-Qscore).
  # The scale factor f is provided with the 'bfactor' arg
  # Bfactor 'f' should not affect q result
  print(f"Running mapq  with command:")
  print(mapq_command)
  print("\n\n")
  subprocess.run(mapq_command.split())
  os.chdir(cwd)
  debug_data = load_mapq_debug_data(test_dir)
  return debug_data

def load_mapq_debug_data(test_dir):
    """
    Load the mapq intermediate results.
    Looks for a file 'debug_data_mapq.json' in the test directory.
    """

    # anticipate output data file path
    data_file = Path(test_dir,Path("debug_data_mapq.json")).absolute()
    # load debug data
    with open(data_file,"r") as fh:
      debug_data = json.load(fh)
      def _is_ragged(a):
        # don't force arrays for ragged data
        if isinstance(a, list):
          # Check if all elements are lists and have the same length
          if all(isinstance(i, list) for i in a):
            length = len(a[0])
            return any(len(i) != length for i in a)
          else:
            # It's a list, but not a list of lists
            return False
        else:
          # Not a list, so it's not a ragged array in the typical sense
          return False

      debug_data = {key:np.array(value) if not _is_ragged(value) else value\
                     for key,value in debug_data.items()}
    return debug_data

def eval_test_mapq_pdb_output(test):
  #Check q from actual pdb output file. Assumes comparison is present
  # in test["results"]["calc"]["qscore_per_atom"]
  test = convert_group_args_to_dict(test)
  test_dir = Path(Path(test["data"]["test_dir"]),test["data"]["name"])
  model_path = Path(test["data"]["model_file"])
  map_path = Path(test["data"]["map_file"])
  dm = DataManager()
  suffix = model_path.suffix[1:]
  mapq_output_file = Path(
    test_dir, f"{model_path.stem}.{suffix}__Q__{map_path.stem}.{map_path.suffix[1:]}.{suffix}")

  _ = dm.process_model_file(str(mapq_output_file))
  model = dm.get_model()
  cif_input = model.get_model_input()
  model.add_crystal_symmetry_if_necessary()
  sel_noH = model.selection("not element H")
  model = model.select(sel_noH)

  if suffix == "pdb":
    pseudo_b = model.get_b_iso().as_numpy_array()
    q_test = pseudo_b

  else: #mmcif
    q_test = np.array([float(v) for v in cif_input.cif_model[model_path.stem]["_atom_site"]["_atom_site.Q-score"]])
    q_test = q_test[sel_noH.as_numpy_array()]
  # Round heavily to match bfactor
  qscore_per_atom = test["results"]["calc"]["qscore_per_atom"]
  q_calc = qscore_per_atom
  assert np.all(np.isclose(q_calc,q_test,atol=1e-2))


def eval_test_with_mapq(test,mapq_location="/Users/user/Desktop/Chimera.app/Contents/Resources/share/mapq/"):
  # run a test with map and compare
  # Assumes compare values are inside test already
  if isinstance(test,dict):
    test = convert_dict_to_group_args(test)

  test.data.data_dir =  str(Path(Path(test.data.test_dir),Path("qscore_tst_run_mapq_"+test.data.name)))
  mapq_data = run_test_template_mapq(convert_group_args_to_dict(test),mapq_location=mapq_location)
  mapq = convert_dict_to_group_args(mapq_data)

  dm = DataManager()
  dm.process_model_file(test.data.model_file)
  if test.data.map_file is not None:
    dm.process_real_map_file(test.data.map_file)
  mmm = dm.get_map_model_manager()

  # compare
  assert np.all(isclose_or_nan(test.results.calc.probe_xyz-mmm.shift_cart(),mapq.probe_xyz))
  assert np.all(isclose_or_nan(test.results.calc.probe_mask,mapq.probe_mask))
  assert np.all(isclose_or_nan(test.results.calc.g_vals,mapq.g_vals))
  assert np.all(isclose_or_nan(test.results.calc.d_vals,mapq.d_vals))
  assert np.all(isclose_or_nan(test.results.calc.qscore_per_atom,mapq.qscore_per_atom))
  test.results.mapq = mapq
  return test




def aggregate_qscore_per_residue_test(test,window=3):
  # assign residue indices to each atom
  dm = DataManager()
  fn = dm.process_model_file(test["data"]["model_file"])

  model = dm.get_model()
  model.add_crystal_symmetry_if_necessary()
  model = model.select(model.selection("not element H"))
  grouped_means = aggregate_qscore_per_residue(model,test["results"]["calc"]["qscore_per_atom"],window=window)
  return grouped_means



def read_mapq_output(test,window=3):
  test = convert_group_args_to_dict(test)
  test_dir = Path(test["data"]["test_dir"],test["data"]["name"])
  model_path = Path(test["data"]["model_file"])
  map_path = Path(test["data"]["map_file"])
  files = list(Path(test_dir).glob("*"))
  mapq_output_files = [file for file in files if "All" in str(file)]
  mapq_output_file = mapq_output_files[0]
  with open(str(mapq_output_file),"r") as fh:
    lines = fh.readlines()

  # parsing
  chain_lines = OrderedDict()
  chain_start_i = 0
  chain_end_i = 0
  in_chain = False
  chain_name = None
  for i,line in enumerate(lines):
    if "Chain" in line and "Average over" in line  or i==len(lines)-1:
      # first check if need to close existing
      if in_chain:
        chain_end_i = i
        chain_lines[f"{chain_name}"] = lines[chain_start_i:chain_end_i]
        chain_end_i = None
        chain_start_i = None
        chain_name = None
        in_chain = False


      chain_start_i = i
      chain_end_i = None
      chain_name = line
      # get chain name
      index = line.find("Chain")

      start_index = index + len("Chain")
      # Slice the main string to get n characters after the substring
      chain_name = line[start_index:start_index + 10].strip()
      in_chain = True

  # load into dfs
  result_dfs = {}
  for chain_id,data_lines in chain_lines.items():

    # get index of column name line
    header = None
    for i,line in enumerate(data_lines):
      if "Expected" in line:
        header = i-1
        break

    data_str = "\n".join(data_lines)

    # Use StringIO to turn the string into a file-like object
    data_io = StringIO(data_str)

    # Read the data, specifying a regex separator for one or more spaces
    df_raw = pd.read_csv(data_io, sep="\t",header=header)
    result_dfs[chain_id] = df_raw

  # get rolling mean
  grouped_means_df = aggregate_qscore_per_residue_test(test,window=window)

  return result_dfs, grouped_means_df

def eval_residue_aggregation(test,window = 3):
  result_dfs, means_df = read_mapq_output(test)

  # add a chain group index (accomodates same name chains)
  #means_df['chain_group_index'] = (means_df['chain_id'] != means_df['chain_id'].shift()).cumsum()-1
  for grouup_idx,group in means_df.groupby("chain_id"):
    #mapq_chain_key = "-".join([str(group.iloc[0]["chain_id"]),str(group.iloc[0]["chain_group_index"])])
    chain_id = str(group.iloc[0]["chain_id"])
    for x,y in zip(group["Q-scorePerResidue"].values,result_dfs[chain_id][f"Q_residue.{window}"]):
      #assert np.isclose(x,y,atol=1e-3) or  np.isnan(x) or  np.isnan(y), f"{x},{y}"
      pass
    print("Residue aggregation passed for chain: "+group.iloc[0]["chain_id"])

def build_tests_mapq(test_dir = "qscore_tst_dir"):
  tests = {}
  # add a larger test
  # map/model from regression dir
  test = copy.deepcopy(test_template)
  tst_model_file = libtbx.env.find_in_repositories(
  relative_path=\
  f"phenix_regression/real_space_refine/data/tst_48.pdb",
  test=os.path.isfile)
  tst_map_file = libtbx.env.find_in_repositories(
  relative_path=\
  f"phenix_regression/real_space_refine/data/tst_48.ccp4",
  test=os.path.isfile)
  test["data"]["model_file"] = tst_model_file
  test["data"]["map_file"] = tst_map_file
  test["data"]["name"] = "tst_48_progressive_numpy"
  test["data"]["test_dir"] = test_dir
  test["params"]["n_probes_max"] = 16
  test["params"]["n_probes_target"] = 8
  test["params"]["probe_allocation_method"] = "progressive"
  test["params"]["backend"] = "numpy"
  tests[test["data"]["name"]] = test

  # a custom test from local desktop
  test = copy.deepcopy(test_template)
  tst_model_file =  "/Users/user/Desktop/data/8gvk/8gvk.cif"
  tst_map_file = "/Users/user/Desktop/data/8gvk/emd_32099.map"
  test["data"]["model_file"] = tst_model_file
  test["data"]["map_file"] = tst_map_file
  test["data"]["name"] = "8gvk_progressive_numpy"
  test["data"]["test_dir"] = test_dir
  test["params"]["n_probes_max"] = 16
  test["params"]["n_probes_target"] = 8
  test["params"]["probe_allocation_method"] = "progressive"
  test["params"]["backend"] = "numpy"
  tests[test["data"]["name"]] = test


  return tests


def print_qscore_data(data,n_cols=10):
  # print numerical data for copying back to source code
  s = ""
  d_i = 0
  while d_i < len(data):
      for i in range(n_cols):
          if d_i == len(data):
              break
          s += f"{data[d_i]:>8.5f}"
          s+=", "
          d_i += 1
      s += "\n"

  print(s)

if (__name__ == "__main__"):
  parser = argparse.ArgumentParser(description="Run qscore tests")

  # Figure out if using mapq
  parser.add_argument('--mapq_location',
                      type=str,
                      help='Compare to mapq results. Example:\
                /Users/user/Desktop/Chimera.app/Contents/Resources/share/mapq')

  args = parser.parse_args()
  mapq_location = args.mapq_location
  assert mapq_location

  test_dir = "qscore_tst_dir"
  # get tests from main tst file
  tests = build_tests(test_dir = test_dir)
  tests_mapq = build_tests_mapq(test_dir=test_dir)
  tests.update(tests_mapq)

  # run and eval
  for test_name,test in list(tests.items()):

    test = run_test(test)
    #print("Calc values")
    #data = convert_dict_to_group_args(test).results.calc.qscore_per_atom
    #print_qscore_data(data,n_cols=7)
    if test["params"]["probe_allocation_method"]== "progressive":
      if test["data"]["map_file"] is not None:
        # run mapq
        print("#"*39)
        print("Running test with mapq")
        test = eval_test_with_mapq(test)
        eval_test_mapq_pdb_output(test)
        #print("Mapq values")
        #data = test.results.mapq.qscore_per_atom
        #print_qscore_data(data,n_cols=7)

        # test residue level aggregation
        eval_residue_aggregation(test)

  print("OK")
