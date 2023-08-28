from __future__ import absolute_import, division, print_function

import hashlib
import six
import sys

import boost_adaptbx.boost.optional # import dependency
import boost_adaptbx.boost.std_pair # import dependency
import boost_adaptbx.boost.python as bp
from six.moves import range
from six.moves import zip
bp.import_ext("scitbx_array_family_flex_ext")
from scitbx_array_family_flex_ext import *
import scitbx_array_family_flex_ext as ext

import scitbx.stl.map # import dependency
import scitbx.random
from scitbx.random import get_random_seed, set_random_seed
from libtbx.str_utils import format_value

if six.PY3:
  from collections.abc import Iterable, Sequence
else:
  from collections import Iterable, Sequence
# Register extension classes that look like a sequence, ie. have a
# length and adressable elements, as a Sequence. Same for Iterable.
for entry in ext.__dict__.values():
  # Only consider types (=classes), not object instances
  if not isinstance(entry, type): continue
  # The Iterable interface means the type contains retrievable items.
  # If the type fulfills this but is not already a known Iterable then
  # register it as such.
  if hasattr(entry, "__getitem__") and not issubclass(entry, Iterable):
    Iterable.register(entry)
  # A Sequence is an Iterable that also has a determinable length.
  if hasattr(entry, "__getitem__") and hasattr(entry, "__len__") \
      and not issubclass(entry, Sequence):
    Sequence.register(entry)

def bool_md5(self):
  return hashlib.md5(self.__getstate__()[1])
bool.md5 = bool_md5

@bp.inject_into(grid)
class _():

  def show_summary(self, f=None):
    if (f is None): f = sys.stdout
    print("origin:", self.origin(), file=f)
    print("last:", self.last(), file=f)
    print("focus:", self.focus(), file=f)
    print("all:", self.all(), file=f)
    return self

def sorted(data, reverse=False, stable=True):
  return data.select(
    sort_permutation(data=data, reverse=reverse, stable=stable))

def as_scitbx_matrix(a):
  assert a.nd() == 2
  assert a.is_0_based()
  assert not a.is_padded()
  import scitbx.matrix
  return scitbx.matrix.rec(tuple(a), a.focus())

def show(a):
  print(as_scitbx_matrix(a).mathematica_form(one_row_per_line=True))

def rows(a):
  assert a.nd() == 2
  assert a.is_0_based()
  assert not a.is_padded()
  nr,nc = a.focus()
  for ir in range(nr):
    yield a[ir*nc:(ir+1)*nc]

def upper_bidiagonal(d, f):
  n = len(d)
  a = double(n*n)
  a.reshape(grid(n,n))
  for i,x in enumerate(d):
    a[i,i] = x
  for i,x in enumerate(f):
    a[i,i+1] = x
  return a

def lower_bidiagonal(d, f):
  n = len(d)
  a = double(n*n)
  a.reshape(grid(n,n))
  for i,x in enumerate(d): a[i,i] = x
  for i,x in enumerate(f): a[i+1,i] = x
  return a

def export_to(target_module_name):
  export_list = [
    "sorted",
    "show",
    "rows",
    "to_list",
    "min_default",
    "max_default",
    "mean_default",
    "select",
    "condense_as_ranges",
    "get_random_seed",
    "random_generator",
    "set_random_seed",
    "random_size_t",
    "random_double",
    "random_bool",
    "random_permutation",
    "random_selection",
    "random_double_point_on_sphere",
    "random_double_unit_quaternion",
    "random_double_r3_rotation_matrix",
    "random_double_r3_rotation_matrix_arvo_1992",
    "random_int_gaussian_distribution",
    "median",
    "py_object",
    "linear_regression",
    "linear_correlation",
    "histogram",
    "weighted_histogram",
    "show_count_stats",
    "permutation_generator",
    "smart_selection",
    "compare_derivatives"]
  target_module = sys.modules[target_module_name]
  g = globals()
  for attr in export_list:
    setattr(target_module, attr, g[attr])

def to_list(array):
  """Workaround for C++ exception handling bugs
     (list(array) involves C++ exceptions)"""
  result = []
  for i in range(array.size()):
    result.append(array[i])
  return result

def min_default(values, default):
  if (values.size() == 0): return default
  return min(values)

def max_default(values, default):
  if (values.size() == 0): return default
  return max(values)

def mean_default(values, default):
  if (values.size() == 0): return default
  return mean(values)

def _format_min(values, format):
  return format_value(
    format=format, value=min_default(values=values, default=None))

def _format_max(values, format):
  return format_value(
    format=format, value=max_default(values=values, default=None))

def _format_mean(values, format):
  return format_value(
    format=format, value=mean_default(values=values, default=None))

@bp.inject_into(ext.min_max_mean_double)
class _():

  def show(self, out=None, prefix="", format="%.6g", show_n=True):
    if out is None: out = sys.stdout
    if show_n:
      print(prefix + "n:", self.n, file=out)
    def f(v):
      return format_value(format=format, value=v)
    print(prefix + "min: ", f(self.min), file=out)
    print(prefix + "max: ", f(self.max), file=out)
    print(prefix + "mean:", f(self.mean), file=out)

  def as_tuple(self):
    return (self.min, self.max, self.mean)

def _min_max_mean_double_init(self):
  return min_max_mean_double(values=self)

def _standard_deviation_helper(data, m):
  den = data.size() - m
  if den <= 0: return None
  return (sum(pow2(data - mean(data))) / den)**0.5

def _standard_deviation_of_the_sample(self):
  return _standard_deviation_helper(self, 0)

def _sample_standard_deviation(self):
  return _standard_deviation_helper(self, 1)

def _rms(data):
  den = data.size()
  if den <= 0: return None
  return (sum(pow2(data)) / den)**0.5

def _as_z_scores(self):
  rc=min_max_mean_double(values=self)
  ssd = _sample_standard_deviation(self)
  self -= rc.mean
  self /= ssd

double.format_min = _format_min
double.format_max = _format_max
double.format_mean = _format_mean
double.min_max_mean = _min_max_mean_double_init
double.standard_deviation_of_the_sample = _standard_deviation_of_the_sample
double.sample_standard_deviation = _sample_standard_deviation
double.rms = _rms
double.as_z_scores = _as_z_scores

def select(sequence, permutation=None, flags=None):
  result = []
  if (permutation is not None):
    assert flags is None
    for i in permutation:
      result.append(sequence[i])
  else:
    assert flags is not None
    for s,f in zip(sequence, flags):
      if (f): result.append(s)
  return result

def condense_as_ranges(integer_array):
  if (len(integer_array) == 0): return []
  result = []
  i_start = integer_array[0]
  n = 1
  def store_range():
    if (n == 1):
      result.append((i_start,))
    else:
      result.append((i_start, i_start+n-1))
  for i in integer_array[1:]:
    if (i == i_start + n):
      n += 1
    else:
      store_range()
      i_start = i
      n = 1
  store_range()
  return result

@bp.inject_into(mersenne_twister)
class _():

  def random_selection(self, population_size, sample_size):
    assert population_size >= 0
    assert sample_size >= 0
    assert sample_size <= population_size
    perm = self.random_permutation(size=population_size)
    perm.resize(sample_size)
    return sorted(perm)

random_generator = ext.mersenne_twister(scitbx.random.mt19937)

def set_random_seed(value):
  random_generator.seed(value=value)
  scitbx.random.set_random_seed(value)

random_size_t = random_generator.random_size_t
random_double = random_generator.random_double
random_bool = random_generator.random_bool
random_permutation = random_generator.random_permutation
random_selection = random_generator.random_selection
random_double_point_on_sphere = random_generator.random_double_point_on_sphere
random_double_unit_quaternion = random_generator.random_double_unit_quaternion
random_double_r3_rotation_matrix \
  = random_generator.random_double_r3_rotation_matrix
random_double_r3_rotation_matrix_arvo_1992 \
  = random_generator.random_double_r3_rotation_matrix_arvo_1992
random_int_gaussian_distribution \
  = random_generator.random_int_gaussian_distribution

median = ext.median_functor(seed=get_random_seed())


class py_object(object):

  def __init__(self, accessor, value=None, values=None, value_factory=None):
    assert [value, values, value_factory].count(None) >= 2
    self._accessor = accessor
    if (value_factory is not None):
      self._data = [value_factory() for i in range(accessor.size_1d())]
    elif (values is not None):
      assert len(values) == accessor.size_1d()
      self._data = values[:]
    else:
      self._data = [value for i in range(accessor.size_1d())]

  def accessor(self):
    return self._accessor

  def data(self):
    return self._data

  def __getitem__(self, index):
    return self._data[self._accessor(index)]

  def __setitem__(self, index, value):
    self._data[self._accessor(index)] = value

@bp.inject_into(ext.linear_regression_core)
class _():

  def show_summary(self, f=None, prefix=""):
    if (f is None): f = sys.stdout
    print(prefix+"is_well_defined:", self.is_well_defined(), file=f)
    print(prefix+"y_intercept:", self.y_intercept(), file=f)
    print(prefix+"slope:", self.slope(), file=f)

@bp.inject_into(ext.double)
class _():

  def matrix_inversion(self):
    result = self.deep_copy()
    result.matrix_inversion_in_place()
    return result

  def as_scitbx_matrix(self):
    return as_scitbx_matrix(self)

@bp.inject_into(ext.linear_correlation)
class _():

  def show_summary(self, f=None, prefix=""):
    if (f is None): f = sys.stdout
    print(prefix+"is_well_defined:", self.is_well_defined(), file=f)
    print(prefix+"mean_x:", self.mean_x(), file=f)
    print(prefix+"mean_y:", self.mean_y(), file=f)
    print(prefix+"coefficient:", self.coefficient(), file=f)

class histogram_slot_info(object):

  def __init__(self, low_cutoff, high_cutoff, n):
    self.low_cutoff = low_cutoff
    self.high_cutoff = high_cutoff
    self.n = n

  def center(self):
    return (self.high_cutoff + self.low_cutoff) / 2

@bp.inject_into(ext.histogram)
class _():

  def __getinitargs__(self):
    return (
      self.data_min(),
      self.data_max(),
      self.slot_width(),
      self.slots(),
      self.n_out_of_slot_range())

  def __str__(self):
    from libtbx.utils import kludge_show_to_str
    return kludge_show_to_str(self)

  def slot_infos(self):
    low_cutoff = self.data_min()
    for i,n in enumerate(self.slots()):
      high_cutoff = self.data_min() + self.slot_width() * (i+1)
      yield histogram_slot_info(low_cutoff, high_cutoff, n)
      low_cutoff = high_cutoff

  def show(self, f=None, prefix="", format_cutoffs="%.8g"):
    if (f is None): f = sys.stdout
    print(self.as_str(prefix=prefix, format_cutoffs=format_cutoffs), file=f)

  def as_str(self, prefix="", format_cutoffs="%.8g"):
    output = []
    fmt = "%s" + format_cutoffs + " - " + format_cutoffs + ": %d"
    for info in self.slot_infos():
      output.append(fmt % (prefix, info.low_cutoff, info.high_cutoff, info.n))
    return "\n".join(output)

def show_count_stats(
      counts,
      group_size=10,
      label_0="None",
      out=None,
      prefix=""):
  assert counts.size() != 0
  if (out is None): out = sys.stdout
  from builtins import int, max
  counts_sorted = sorted(counts, reverse=True)
  threshold = max(1, int(counts_sorted[0] / group_size) * group_size)
  n = counts_sorted.size()
  wt = max(len(label_0), len(str(threshold)))
  wc = len(str(n))
  fmt_val  = prefix + ">= %%%dd:  %%%dd  %%7.5f" % (wt, wc)
  fmt_zero = prefix + "   %s:  %%%dd  %%7.5f" % (("%%%ds" % wt) % label_0, wc)
  for i,count in enumerate(counts_sorted):
    if (count >= threshold): continue
    assert count >= 0
    if (i > 0):
      print(fmt_val % (threshold, i, i/n), file=out)
    if (count == 0):
      print(fmt_zero % (n-i, 1-i/n), file=out)
      break
    threshold = max(1, threshold-group_size)
  else:
    print(fmt_val % (threshold, n, 1), file=out)

class weighted_histogram_slot_info(object):

  def __init__(self, low_cutoff, high_cutoff, n):
    self.low_cutoff = low_cutoff
    self.high_cutoff = high_cutoff
    self.n = n

  def center(self):
    return (self.high_cutoff + self.low_cutoff) / 2

@bp.inject_into(ext.weighted_histogram)
class _():

  def __getinitargs__(self):
    return (
      self.data_min(),
      self.data_max(),
      self.slot_width(),
      self.slots(),
      self.n_out_of_slot_range())

  def slot_infos(self):
    low_cutoff = self.data_min()
    for i,n in enumerate(self.slots()):
      high_cutoff = self.data_min() + self.slot_width() * (i+1)
      yield weighted_histogram_slot_info(low_cutoff, high_cutoff, n)
      low_cutoff = high_cutoff

  def show(self, f=None, prefix="", format_cutoffs="%.8g"):
    if (f is None): f = sys.stdout
    fmt = "%s" + format_cutoffs + " - " + format_cutoffs + ": %d"
    for info in self.slot_infos():
      print(fmt % (prefix, info.low_cutoff, info.high_cutoff, info.n), file=f)

def permutation_generator(size):
  result = size_t(range(size))
  yield result
  while (result.next_permutation()): yield result

class smart_selection(object):

  bool_element_size = bool.element_size()
  size_t_element_size = size_t.element_size()

  def __init__(self, flags=None, indices=None, all_size=None):
    "Self-consistency of flags, indices, all_size is not checked!"
    self._flags = flags
    self._indices = indices
    self._all_size = all_size
    self._selected_size = None

  def _get_all_size(self):
    if (    self._all_size is None
        and self._flags is not None):
      self._all_size = self._flags.size()
    return self._all_size
  all_size = property(_get_all_size)

  def _get_selected_size(self):
    if (self._selected_size is None):
      if (self._indices is not None):
        self._selected_size = self._indices.size()
      elif (self._flags is not None):
        self._selected_size = self._flags.count(True)
    return self._selected_size
  selected_size = property(_get_selected_size)

  def _get_flags(self):
    if (    self._flags is None
        and self._all_size is not None
        and self._indices is not None):
      self._flags = bool(self._all_size, self._indices)
    return self._flags
  flags = property(_get_flags)

  def _get_indices(self):
    if (    self._indices is None
        and self._flags is not None):
      self._indices = self._flags.iselection()
    return self._indices
  indices = property(_get_indices)

  def __eq__(self, other):
    if (self.all_size != other.all_size): return False
    if (self._flags is not None):
      return self._flags.all_eq(other.flags)
    if (self._indices is not None):
      return self._indices.all_eq(other.indices)
    return True

  def __getstate__(self):
    asz = self.all_size
    if (asz is None):
      return (None, None, None, self._selected_size)
    if (asz == 0):
      return (None, self.indices, 0, self._selected_size)
    ssz = self.selected_size
    if (  asz * self.bool_element_size
        < ssz * self.size_t_element_size):
      return (self.flags, None, asz, self._selected_size)
    return (None, self.indices, asz, self._selected_size)

  def __setstate__(self, state):
    self._flags, self._indices, self._all_size, self._selected_size = state

  def format_summary(self):
    asz = self.all_size
    if (asz is None):
      idc = self.indices
      if (idc is None): return "None"
      return "%d" % idc.size()
    ssz = self.selected_size
    if (ssz == asz):
      if (ssz == 0): return "None (empty array)"
      return "all (%d)" % ssz
    return "%d of %d" % (ssz, asz)

  def show_summary(self, out=None, prefix="", label="selected elements: "):
    if (out is None): out = sys.stdout
    print(prefix + label + self.format_summary(), file=out)


def __show_sizes(f):
  typename_n_size = f()
  from builtins import max
  l = max([ len(typename) for typename, size in typename_n_size ])
  fmt = "%%%is : %%i" % l
  for typename, size in typename_n_size:
    print(fmt % (typename, size))

show_sizes_int = lambda: __show_sizes(empty_container_sizes_int)
show_sizes_double = lambda: __show_sizes(empty_container_sizes_double)

def exercise_triple(flex_triple, flex_order=None, as_double=False):
  from libtbx.test_utils import approx_equal
  from six.moves import cPickle as pickle
  a = flex_triple()
  assert a.size() == 0
  a = flex_triple(132)
  assert a.size() == 132
  for x in a:
    assert x == (0,0,0)
  a = flex_triple(((1,2,3), (2,3,4), (3,4,5)))
  assert a.size() == 3
  assert tuple(a) == ((1,2,3), (2,3,4), (3,4,5))
  p = pickle.dumps(a)
  b = pickle.loads(p)
  assert tuple(a) == tuple(b)
  if (flex_order is not None):
    assert flex_order(a, b) == 0
  if (as_double):
    assert approx_equal(tuple(a.as_double()), (1,2,3,2,3,4,3,4,5))
    b = flex_triple(a.as_double())
    assert tuple(a) == tuple(b)

def compare_derivatives(more_reliable, less_reliable, eps=1e-6):
  from builtins import max
  scale = max(1, ext.max(ext.abs(more_reliable)))
  if (not (more_reliable/scale).all_approx_equal( # fast
             other=less_reliable/scale, tolerance=eps)):
    from libtbx.test_utils import approx_equal
    assert approx_equal( # slow but helpful output
      more_reliable/scale, less_reliable/scale, eps=eps)

def sum(flex_array, axis=None):
  """ Support for numpy-style summation along an axis.
      If axis=None then summation is performed over the entire array.
  """
  if axis is None:
    return ext.sum(flex_array)
  elif flex_array.nd() == 1:
    assert axis == 0
    return ext.sum(flex_array)
  else:
    old_dim = list(flex_array.all())
    assert axis < len(old_dim)
    new_dim = list(flex_array.all())
    new_dim.pop(axis)
    flex_array_sum = flex_array.__class__(grid(new_dim), 0)
    slices = [slice(0, old_dim[i]) for i in range(len(old_dim))]
    for i in range(old_dim[axis]):
      slices[axis] = slice(i,i+1)
      flex_array_sum += flex_array[slices]
    return flex_array_sum

def _vec3_double_as_numpy_array(flex_array):
  """
  A short extension method for converting vec3_double arrays to numpy arrays.
  """
  if isinstance(flex_array, type(vec3_double())):
    return flex_array.as_double().as_numpy_array().reshape(-1, 3)

vec3_double.as_numpy_array = _vec3_double_as_numpy_array

# for modern 64-bit platforms, int and int32_t are the same
int32 = ext.int
int32_from_byte_str = ext.int_from_byte_str
int32_range = ext.int_range

# int64_t is the same as long, but not on Windows
if sys.platform != 'win32':
  int64 = ext.long
  int64_from_byte_str = ext.long_from_byte_str
  int64_range = ext.long_range

# uint64_t is the same as size_t
uint64 = ext.size_t
uint64_from_byte_str = ext.size_t_from_byte_str
uint64_range = ext.size_t_range