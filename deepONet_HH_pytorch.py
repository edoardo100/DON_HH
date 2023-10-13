""" Utilities slightly modified from DeepXDE classes
CREDITS:

    @article{lu2021deepxde,
      author  = {Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
      title   = {{DeepXDE}: A deep learning library for solving differential equations},
      journal = {SIAM Review},
      volume  = {63},
      number  = {1},
      pages   = {208-228},
      year    = {2021},
      doi     = {10.1137/19M1274067}
    }
"""
import torch
import torch.nn.functional as F
import abc
import numpy as np

mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## Metal GPU acceleration on Mac OSX
## NOT WORKING since ComplexFloat is not supported by the MPS backend
#if torch.backends.mps.is_available():
#    mydevice = torch.device("mps")
#    print ("MPS acceleration is enabled")
#else:
#    print ("MPS device not found.")
torch.set_default_device(mydevice) # default tensor device
torch.set_default_tensor_type(torch.FloatTensor) # default tensor dtype

class BatchSampler:
    """Samples a mini-batch of indices.

    The indices are repeated indefinitely. Has the same effect as:

    .. code-block:: python

        indices = tf.data.Dataset.range(num_samples)
        indices = indices.repeat().shuffle(num_samples).batch(batch_size)
        iterator = iter(indices)
        batch_indices = iterator.get_next()

    However, ``tf.data.Dataset.__iter__()`` is only supported inside of ``tf.function`` or when eager execution is
    enabled. ``tf.data.Dataset.make_one_shot_iterator()`` supports graph mode, but is too slow.

    This class is not implemented as a Python Iterator, so that it can support dynamic batch size.

    Args:
        num_samples (int): The number of samples.
        shuffle (bool): Set to ``True`` to have the indices reshuffled at every epoch.
    """

    def __init__(self, num_samples, shuffle=True):
        self.num_samples = num_samples
        self.shuffle = shuffle

        self._indices = np.arange(self.num_samples)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # Shuffle for the first epoch
        if shuffle:
            np.random.shuffle(self._indices)

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_next(self, batch_size):
        """Returns the indices of the next batch.

        Args:
            batch_size (int): The number of elements to combine in a single batch.
        """
        if batch_size > self.num_samples:
            raise ValueError(
                "batch_size={} is larger than num_samples={}.".format(
                    batch_size, self.num_samples
                )
            )

        start = self._index_in_epoch
        if start + batch_size <= self.num_samples:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._indices[start:end]
        else:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_samples = self.num_samples - start
            indices_rest_part = np.copy(
                self._indices[start : self.num_samples]
            )  # self._indices will be shuffled below.
            # Shuffle the indices
            if self.shuffle:
                np.random.shuffle(self._indices)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_samples
            end = self._index_in_epoch
            indices_new_part = self._indices[start:end]
            return np.hstack((indices_rest_part, indices_new_part))

class Data(abc.ABC):
    """Data base class."""

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """Return a list of losses, i.e., constraints."""
        raise NotImplementedError("Data.losses is not implemented.")

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """Return a list of losses for training dataset, i.e., constraints."""
        return self.losses(targets, outputs, loss_fn, inputs, model, aux=aux)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """Return a list of losses for test dataset, i.e., constraints."""
        return self.losses(targets, outputs, loss_fn, inputs, model, aux=aux)

    @abc.abstractmethod
    def train_next_batch(self, batch_size=None):
        """Return a training dataset of the size `batch_size`."""

    @abc.abstractmethod
    def test(self):
        """Return a test dataset."""


######### NN-related classes
class NN(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()
        self.regularizer = None
        self._auxiliary_vars = None
        self._input_transform = None
        self._output_transform = None

    @property
    def auxiliary_vars(self):
        """Tensors: Any additional variables needed."""
        return self._auxiliary_vars

    @auxiliary_vars.setter
    def auxiliary_vars(self, value):
        self._auxiliary_vars = value

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

class FNN(NN):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes, activation, kernel_initializer):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations, activation))
        else:
            self.activation = activations(activation)
        initializer = initializers(kernel_initializer)
        initializer_zero = initializers("zeros")

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

######### Activation functions and initializers
def layer_wise_locally_adaptive(activation, n=1):
    """Layer-wise locally adaptive activation functions (L-LAAF).

    Examples:

    To define a L-LAAF ReLU with the scaling factor ``n = 10``:

    .. code-block:: python

        n = 10
        activation = f"LAAF-{n} relu"  # "LAAF-10 relu"

    References:
        `A. D. Jagtap, K. Kawaguchi, & G. E. Karniadakis. Locally adaptive activation
        functions with slope recovery for deep and physics-informed neural networks.
        Proceedings of the Royal Society A, 476(2239), 20200334, 2020
        <https://doi.org/10.1098/rspa.2020.0334>`_.
    """
    a = torch.nn.parameter.Parameter(torch.Tensor(1 / n))
    return lambda x: activation(n * a * x)

def activations(act_fun,adapt_actfun=False):
    act_dict = {
        "ReLu" : F.relu,
        "tanh" : F.tanh,
    }
    if adapt_actfun:
        return layer_wise_locally_adaptive(act_dict[act_fun],n=10)
    return act_dict[act_fun]
    
def initializers(initial):
    initial_dict = {
        "Glorot normal": torch.nn.init.xavier_normal_,
        "Glorot uniform": torch.nn.init.xavier_uniform_,
        "He normal": torch.nn.init.kaiming_normal_,
        "He uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    return initial_dict[initial]

######### DeepONet-related classes

class TripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        if len(X_train[0]) * len(X_train[1]) != y_train.size:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if len(X_test[0]) * len(X_test[1]) != y_test.size:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]

    def test(self):
        return self.test_x, self.test_y
    
class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations(activation["branch"])
            self.activation_trunk = activations(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations(activation)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)
        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x