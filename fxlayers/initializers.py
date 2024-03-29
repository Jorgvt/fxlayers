# AUTOGENERATED! DO NOT EDIT! File to edit: ../Notebooks/01_initializers.ipynb.

# %% auto 0
__all__ = ['bounded_uniform', 'displaced_normal', 'freq_scales_init', 'k_array', 'log_k_array', 'linspace', 'equal_to', 'mean']

# %% ../Notebooks/01_initializers.ipynb 3
from jax import random, numpy as jnp
from jax._src import dtypes

# %% ../Notebooks/01_initializers.ipynb 5
def bounded_uniform(minval=0.0,
                    maxval=1.0,
                    dtype=dtypes.float_,
                    ):
  def init(key,
           shape,
           dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.uniform(key, shape, dtype, minval, maxval)
  return init

# %% ../Notebooks/01_initializers.ipynb 9
def displaced_normal(mean=0., # Mean of the distribution.
                     stddev=1e-2, # Standard deviation of the distribution.
                     dtype=dtypes.float_ # Desired DType of the resulting array.
                     ):
  """Builds an initializer that returns real normally-distributed random arrays."""

  def init(key,
           shape,
           dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return mean + random.normal(key, shape, dtype) * stddev
  return init

# %% ../Notebooks/01_initializers.ipynb 12
def freq_scales_init(n_scales, # Number of scales.
                     fs, # Sampling frequency.
                     dtype=dtypes.float_ # Desired DType of the resulting array.
                     ):
  """"""

  def init(key,
           shape,
           dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    e = jnp.arange(start=1, stop=n_scales+1)
    fM = fs/(2**e)
    return fM - (fM-fM/2)/2
  return init

# %% ../Notebooks/01_initializers.ipynb 14
def k_array(k, # Number of scales.
            arr, # Sampling frequency.
            dtype=dtypes.float_ # Desired DType of the resulting array.
            ):
  """"""

  def init(key,
           shape,
           dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return k/arr
  return init

# %% ../Notebooks/01_initializers.ipynb 16
def log_k_array(k, # Number of scales.
            arr, # Sampling frequency.
            dtype=dtypes.float_ # Desired DType of the resulting array.
            ):
  """Initializer that generates the weights based on applying the log to a given array."""

  def init(key,
           shape,
           dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.log(k*arr)
  return init

# %% ../Notebooks/01_initializers.ipynb 18
def linspace(start,
             stop,
             num,
             dtype=dtypes.float_ # Desired DType of the resulting array.
             ):
  """"""

  def init(key,
           shape,
           dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.linspace(start=start, stop=stop, num=num+1, dtype=dtype)[:-1]
  return init

# %% ../Notebooks/01_initializers.ipynb 20
def equal_to(arr,
             dtype=dtypes.float_ # Desired DType of the resulting array.
             ):
  """"""

  def init(key,
           shape,
           dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.array(arr, dtype=dtype)
  return init

# %% ../Notebooks/01_initializers.ipynb 22
def mean(dtype=dtypes.float_ # Desired DType of the resulting array.
              ):
  """Builds an initializer that returns a kernel that calculates the mean of the interacting pixels."""

  def init(key,
           shape,
           dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.ones(shape, dtype)/jnp.prod(jnp.array(shape))
  return init
