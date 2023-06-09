# AUTOGENERATED! DO NOT EDIT! File to edit: ../Notebooks/00_layers.ipynb.

# %% auto 0
__all__ = ['GaussianLayer', 'GaborLayer', 'CenterSurroundLogSigma', 'CenterSurroundLogSigmaK']

# %% ../Notebooks/00_layers.ipynb 3
import jax
from typing import Any, Callable, Sequence, Union
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax
from einops import rearrange

from .initializers import bounded_uniform, displaced_normal

# %% ../Notebooks/00_layers.ipynb 7
class GaussianLayer(nn.Module):
    """Parametric gaussian layer."""
    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1 # Sampling frequency
    normalize_prob: bool = True
    normalize_energy: bool = False

    @nn.compact
    def __call__(self,
                 inputs,
                 train=False,
                 ):
        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable("precalc_filter",
                                        "kernel",
                                        jnp.zeros,
                                        (self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
        sigma = self.param("sigma",
                           nn.initializers.uniform(scale=self.xmean),
                           (self.features*inputs.shape[-1],))
        A = self.param("A",
                       nn.initializers.ones,
                       (self.features*inputs.shape[-1],))
        if self.use_bias: bias = self.param("bias",
                                            self.bias_init,
                                            (self.features,))
        else: bias = 0.

        if is_initialized and not train: 
            kernel = precalc_filters.value
        elif is_initialized and train: 
            x, y = self.generate_dominion()
            kernel = jax.vmap(self.gaussian, in_axes=(None,None,None,None,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, sigma, A, self.normalize_prob, self.normalize_energy)
            # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
            kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4: inputs = inputs[None,:]; had_batch = False
        else: had_batch = True
        outputs = lax.conv(jnp.transpose(inputs,[0,3,1,2]),    # lhs = NCHW image tensor
               jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
               (self.strides, self.strides),
               self.padding)
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0,2,3,1))
        if not had_batch: outputs = outputs[0]
        return outputs + bias

    @staticmethod
    def gaussian(x, y, xmean, ymean, sigma, A=1, normalize_prob=True, normalize_energy=False):
        # A_norm = 1/(2*jnp.pi*sigma) if normalize_prob else 1.
        A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*sigma**2), 1.)
        gaussian = A_norm*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2))
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(gaussian**2)), 1.)
        return A*gaussian/E_norm

    def return_kernel(self, params, c_in):
        x, y = self.generate_dominion()
        kernel = jax.vmap(self.gaussian, in_axes=(None,None,None,None,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, params["params"]["sigma"], params["params"]["A"], self.normalize_prob, self.normalize_energy)
        # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, 3, self.features))
        kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=c_in, c_out=self.features)
        return kernel
    
    def generate_dominion(self):
        return jnp.meshgrid(jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1], jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1])

# %% ../Notebooks/00_layers.ipynb 8
class GaussianLayerLogSigma(nn.Module):
    """Parametric gaussian layer that optimizes log(sigma) instead of sigma."""
    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1 # Sampling frequency
    normalize_prob: bool = True
    normalize_energy: bool = False

    @nn.compact
    def __call__(self,
                 inputs,
                 train=False,
                 ):
        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable("precalc_filter",
                                        "kernel",
                                        jnp.zeros,
                                        (self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
        logsigma = self.param("logsigma",
                           bounded_uniform(minval=-4., maxval=-0.5),
                           (self.features*inputs.shape[-1],))
        A = self.param("A",
                       nn.initializers.ones,
                       (self.features*inputs.shape[-1],))
        sigma = jnp.exp(logsigma)
        if self.use_bias: bias = self.param("bias",
                                            self.bias_init,
                                            (self.features,))
        else: bias = 0.
        if is_initialized and not train: 
            kernel = precalc_filters.value
        elif is_initialized and train: 
            x, y = self.generate_dominion()
            kernel = jax.vmap(self.gaussian, in_axes=(None,None,None,None,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, sigma, A, self.normalize_prob, self.normalize_energy)
            # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
            kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4: inputs = inputs[None,:]; had_batch = False
        else: had_batch = True
        outputs = lax.conv(jnp.transpose(inputs,[0,3,1,2]),    # lhs = NCHW image tensor
               jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
               (self.strides, self.strides),
               self.padding)
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0,2,3,1))
        if not had_batch: outputs = outputs[0]
        return outputs + bias

    @staticmethod
    def gaussian(x, y, xmean, ymean, sigma, A=1, normalize_prob=True, normalize_energy=False):
        # A_norm = 1/(2*jnp.pi*sigma) if normalize_prob else 1.
        A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*sigma**2), 1.)
        g = A_norm*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2))
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(g**2)), 1.)
        return A*g/E_norm

    def return_kernel(self, params, c_in):
        x, y = self.generate_dominion()
        kernel = jax.vmap(self.gaussian, in_axes=(None,None,None,None,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, jnp.exp(params["params"]["logsigma"]), params["params"]["A"], self.normalize_prob, self.normalize_energy)
        # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, 3, self.features))
        kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=c_in, c_out=self.features)
        return kernel
    
    def generate_dominion(self):
        return jnp.meshgrid(jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1], jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1])

# %% ../Notebooks/00_layers.ipynb 20
class GaborLayer(nn.Module):
    """Parametric Gabor layer."""
    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1 # Sampling frequency

    normalize_prob: bool = True
    normalize_energy: bool = False

    @nn.compact
    def __call__(self,
                 inputs,
                 train=False,
                 ):
        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable("precalc_filter",
                                        "kernel",
                                        jnp.zeros,
                                        (self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
        freq = self.param("freq",
                           nn.initializers.uniform(scale=self.fs/2),
                           (self.features*inputs.shape[-1],))
        logsigmax = self.param("logsigmax",
                           bounded_uniform(minval=-4., maxval=-0.5),
                           (self.features*inputs.shape[-1],))
        logsigmay = self.param("logsigmay",
                           bounded_uniform(minval=-4., maxval=-0.5),
                           (self.features*inputs.shape[-1],))        
        theta = self.param("theta",
                           nn.initializers.uniform(scale=jnp.pi),
                           (self.features*inputs.shape[-1],))
        sigma_theta = self.param("sigma_theta",
                           nn.initializers.uniform(scale=jnp.pi),
                           (self.features*inputs.shape[-1],))
        rot_theta = self.param("rot_theta",
                           nn.initializers.uniform(scale=jnp.pi),
                           (self.features*inputs.shape[-1],))
        A = self.param("A",
                       nn.initializers.ones,
                       (self.features*inputs.shape[-1],))
        sigmax, sigmay = jnp.exp(logsigmax), jnp.exp(logsigmay)
        if self.use_bias: bias = self.param("bias",
                                            self.bias_init,
                                            (self.features,))
        else: bias = 0.
        if is_initialized and not train: 
            kernel = precalc_filters.value
        elif is_initialized and train: 
            x, y = self.generate_dominion()
            # gabor_fn = jax.vmap(self.gabor, in_axes=(None,None,None,None,0,0,0,0,0,0,None,None))
            kernel = jax.vmap(self.gabor, in_axes=(None,None,None,None,0,0,0,0,0,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, sigmax, sigmay, freq, theta, sigma_theta, rot_theta, A, self.normalize_prob, self.normalize_energy)
            kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
            # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4: inputs = inputs[None,:]; had_batch = False
        else: had_batch = True
        outputs = lax.conv(jnp.transpose(inputs,[0,3,1,2]),    # lhs = NCHW image tensor
               jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
               (self.strides, self.strides),
               self.padding)
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0,2,3,1))
        if not had_batch: outputs = outputs[0]
        return outputs + bias

    @staticmethod
    def gabor(x, y, xmean, ymean, sigmax, sigmay, freq, theta, sigma_theta, rot_theta, A=1, normalize_prob=True, normalize_energy=False):
        # ## Rotate the dominion
        # x = jnp.cos(rot_theta) * (x - xmean) - jnp.sin(rot_theta) * (y - ymean)
        # y = jnp.sin(rot_theta) * (x - xmean) + jnp.cos(rot_theta) * (y - ymean)
        x, y = x-xmean, y-ymean
        ## Obtain the normalization coeficient
        sigma_vector = jnp.array([sigmax, sigmay])
        cov_matrix = jnp.diag(sigma_vector)**2
        det_cov_matrix = jnp.linalg.det(cov_matrix)
        # A_norm = 1/(2*jnp.pi*jnp.sqrt(det_cov_matrix)) if normalize_prob else 1.
        A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*jnp.sqrt(det_cov_matrix)), 1.)
        
        ## Rotate the sinusoid
        rotation_matrix = jnp.array([[jnp.cos(sigma_theta), -jnp.sin(sigma_theta)],
                                     [jnp.sin(sigma_theta), jnp.cos(sigma_theta)]])
        rotated_covariance = rotation_matrix @ jnp.linalg.inv(cov_matrix) @ jnp.transpose(rotation_matrix)
        x_r_1 = rotated_covariance[0,0] * x + rotated_covariance[0,1] * y
        y_r_1 = rotated_covariance[1,0] * x + rotated_covariance[1,1] * y
        distance = x * x_r_1 + y * y_r_1
        g = A_norm*jnp.exp(-distance/2) * jnp.cos(2*jnp.pi*freq*(x*jnp.cos(theta)+y*jnp.sin(theta)))
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(g**2)), 1.)
        return A*g/E_norm

    def return_kernel(self, params, c_in=3):
        x, y = self.generate_dominion()
        sigmax, sigmay = jnp.exp(params["logsigmax"]), jnp.exp(params["logsigmay"])
        # sigmax, sigmay = jnp.exp(params["sigmax"]), jnp.exp(params["sigmay"])
        kernel = jax.vmap(self.gabor, in_axes=(None,None,None,None,0,0,0,0,0,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, sigmax, sigmay, params["freq"], params["theta"], params["sigma_theta"], params["rot_theta"], params["A"], self.normalize_prob, self.normalize_energy)
        # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, input_channels, self.features))
        kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=c_in, c_out=self.features)
        return kernel
    
    def generate_dominion(self):
        return jnp.meshgrid(jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1], jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1])

# %% ../Notebooks/00_layers.ipynb 33
class CenterSurroundLogSigma(nn.Module):
    """Parametric center surround layer that optimizes log(sigma) instead of sigma."""
    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1 # Sampling frequency
    normalize_prob: bool = True
    normalize_energy: bool = False

    @nn.compact
    def __call__(self,
                 inputs,
                 train=False,
                 ):
        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable("precalc_filter",
                                        "kernel",
                                        jnp.zeros,
                                        (self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
        logsigma = self.param("logsigma",
                           bounded_uniform(minval=-2.2, maxval=-1.7),
                           (self.features*inputs.shape[-1],))
        logsigma2 = self.param("logsigma2",
                           bounded_uniform(minval=-2.2, maxval=-1.7),
                           (self.features*inputs.shape[-1],))
        A = self.param("A",
                       nn.initializers.ones,
                       (self.features*inputs.shape[-1],))
        sigma = jnp.exp(logsigma)
        sigma2 = jnp.exp(logsigma2)
        if self.use_bias: bias = self.param("bias",
                                            self.bias_init,
                                            (self.features,))
        else: bias = 0.
        if is_initialized and not train: 
            kernel = precalc_filters.value
        elif is_initialized and train: 
            x, y = self.generate_dominion()
            kernel = jax.vmap(self.center_surround, in_axes=(None,None,None,None,0,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, sigma, sigma2, A, self.normalize_prob, self.normalize_energy)
            # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
            kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4: inputs = inputs[None,:]; had_batch = False
        else: had_batch = True
        outputs = lax.conv(jnp.transpose(inputs,[0,3,1,2]),    # lhs = NCHW image tensor
               jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
               (self.strides, self.strides),
               self.padding)
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0,2,3,1))
        if not had_batch: outputs = outputs[0]
        return outputs + bias

    # @staticmethod
    # def gaussian(x, y, xmean, ymean, sigma, A=1, normalize_prob=True):
    #     # A_norm = 1/(2*jnp.pi*sigma) if normalize_prob else 1.
    #     A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*sigma), 1.)
    #     return A*A_norm*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2))
    
    @staticmethod
    def center_surround(x, y, xmean, ymean, sigma, sigma2, A=1, normalize_prob=True, normalize_energy=False):
        def gaussian(x, y, xmean, ymean, sigma, A=1, normalize_prob=True):
            A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*sigma**2), 1.)
            return A*A_norm*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2))
        g1 = gaussian(x, y, xmean, ymean, sigma, 1, normalize_prob)
        g2 = gaussian(x, y, xmean, ymean, sigma2, 1, normalize_prob)
        g = g1-g2
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(g**2)), 1.)
        return A*g/E_norm
    
    # @staticmethod
    # def center_surround(x, y, xmean, ymean, sigma,  K, A=1, normalize_prob=True):
    #     return (1/(2*jnp.pi*sigma**2))*(jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2)) - (1/(K**2))*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*(K*sigma)**2)))

    def return_kernel(self, params, c_in):
        x, y = self.generate_dominion()
        kernel = jax.vmap(self.center_surround, in_axes=(None,None,None,None,0,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, jnp.exp(params["params"]["logsigma"]), jnp.exp(params["params"]["logsigma2"]), params["params"]["A"], self.normalize_prob, self.normalize_energy)
        # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, 3, self.features))
        kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=c_in, c_out=self.features)
        return kernel
    
    def generate_dominion(self):
        return jnp.meshgrid(jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1], jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1])

# %% ../Notebooks/00_layers.ipynb 36
class CenterSurroundLogSigmaK(nn.Module):
    """Parametric center surround layer that optimizes log(sigma) instead of sigma and has a factor K instead of a second sigma."""
    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: int = 1
    padding: str = "SAME"
    feature_group_count: int = 1
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    use_bias: bool = False
    xmean: float = 0.5
    ymean: float = 0.5
    fs: float = 1 # Sampling frequency
    normalize_prob: bool = True
    normalize_energy: bool = True

    @nn.compact
    def __call__(self,
                 inputs,
                 train=False,
                 ):
        is_initialized = self.has_variable("precalc_filter", "kernel")
        precalc_filters = self.variable("precalc_filter",
                                        "kernel",
                                        jnp.zeros,
                                        (self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
        logsigma = self.param("logsigma",
                           bounded_uniform(minval=-2.2, maxval=-1.7),
                           (self.features*inputs.shape[-1],))
        K = self.param("K",
                           displaced_normal(mean=1.1, stddev=0.1),
                           (self.features*inputs.shape[-1],))
        A = self.param("A",
                       nn.initializers.ones,
                       (self.features*inputs.shape[-1],))
        sigma = jnp.exp(logsigma)
        sigma2 = K*sigma
        if self.use_bias: bias = self.param("bias",
                                            self.bias_init,
                                            (self.features,))
        else: bias = 0.
        if is_initialized and not train: 
            kernel = precalc_filters.value
        elif is_initialized and train: 
            x, y = self.generate_dominion()
            kernel = jax.vmap(self.center_surround, in_axes=(None,None,None,None,0,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, sigma, sigma2, A, self.normalize_prob, self.normalize_energy)
            # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, inputs.shape[-1], self.features))
            kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=inputs.shape[-1], c_out=self.features)
            precalc_filters.value = kernel
        else:
            kernel = precalc_filters.value

        ## Add the batch dim if the input is a single element
        if jnp.ndim(inputs) < 4: inputs = inputs[None,:]; had_batch = False
        else: had_batch = True
        outputs = lax.conv(jnp.transpose(inputs,[0,3,1,2]),    # lhs = NCHW image tensor
               jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
               (self.strides, self.strides),
               self.padding)
        ## Move the channels back to the last dim
        outputs = jnp.transpose(outputs, (0,2,3,1))
        if not had_batch: outputs = outputs[0]
        return outputs + bias

    # @staticmethod
    # def gaussian(x, y, xmean, ymean, sigma, A=1, normalize_prob=True):
    #     # A_norm = 1/(2*jnp.pi*sigma) if normalize_prob else 1.
    #     A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*sigma), 1.)
    #     return A*A_norm*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2))
    
    @staticmethod
    def center_surround(x, y, xmean, ymean, sigma, sigma2, A=1, normalize_prob=True, normalize_energy=False):
        def gaussian(x, y, xmean, ymean, sigma, A=1, normalize_prob=True):
            A_norm = jnp.where(normalize_prob, 1/(2*jnp.pi*sigma**2), 1.)
            return A*A_norm*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2))
        g1 = gaussian(x, y, xmean, ymean, sigma, 1, normalize_prob)
        g2 = gaussian(x, y, xmean, ymean, sigma2, 1, normalize_prob)
        g = g1 - g2
        E_norm = jnp.where(normalize_energy, jnp.sqrt(jnp.sum(g**2)), 1.)
        return A*g/E_norm
    
    # @staticmethod
    # def center_surround(x, y, xmean, ymean, sigma,  K, A=1, normalize_prob=True):
    #     return (1/(2*jnp.pi*sigma**2))*(jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*sigma**2)) - (1/(K**2))*jnp.exp(-((x-xmean)**2 + (y-ymean)**2)/(2*(K*sigma)**2)))

    def return_kernel(self, params, c_in):
        x, y = self.generate_dominion()
        kernel = jax.vmap(self.center_surround, in_axes=(None,None,None,None,0,0,0,None,None), out_axes=0)(x, y, self.xmean, self.ymean, jnp.exp(params["params"]["logsigma"]), params["params"]["K"]*jnp.exp(params["params"]["logsigma"]), params["params"]["A"], self.normalize_prob, self.normalize_energy)
        # kernel = jnp.reshape(kernel, newshape=(self.kernel_size, self.kernel_size, 3, self.features))
        kernel = rearrange(kernel, "(c_in c_out) kx ky -> kx ky c_in c_out", c_in=c_in, c_out=self.features)
        return kernel
    
    def generate_dominion(self):
        return jnp.meshgrid(jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1], jnp.linspace(0,self.kernel_size/self.fs,num=self.kernel_size+1)[:-1])
