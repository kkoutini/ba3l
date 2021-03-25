import inspect
import warnings

from pytorch_lightning import Trainer as plTrainer
import inspect
import os

from ba3l.util.functions import get_default_kwargs_dict
from sacred.config.utils import CMD
from .ingredients.datasets import Datasets, raise_
from .ingredients.models import Models
from .ingredients.ingredient import Ingredient

from typing import Sequence, Optional, List

from sacred.utils import PathType, optional_kwargs_decorator

try:
    # loading for pyTorch 1.3
    from torch.utils.data import IterableDataset
except ImportError:
    # loading for pyTorch 1.1
    import torch

    warnings.warn(
        "Your version of pyTorch %s does not support `IterableDataset`,"
        " please upgrade to 1.2+" % torch.__version__,
        ImportWarning,
    )
    EXIST_ITER_DATASET = False
else:
    EXIST_ITER_DATASET = True

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


class Trainer(plTrainer, Ingredient):
    __instance = None

    @classmethod
    def get_instance(cls, datasets=None, models=None):
        if Trainer.__instance is None:
            Trainer.__instance = Trainer(datasets=datasets, models=models)
        return Trainer.__instance

    def __init__(
        self,
        datasets: Datasets = None,
        models: Models = None,
        name: Optional[str] = None,
        ingredients: Sequence[Ingredient] = (),
        interactive: bool = False,
        base_dir: Optional[PathType] = None,
        save_git_info: bool = True,
    ):
        """
        Create a new experiment with the given name and optional ingredients.

        Parameters
        ----------
        name
            Optional name of this experiment, defaults to the filename.
            (Required in interactive mode)

        ingredients : list[sacred.Ingredient], optional
            A list of ingredients to be used with this experiment.

        interactive
            If set to True will allow the experiment to be run in interactive
            mode (e.g. IPython or Jupyter notebooks).
            However, this mode is discouraged since it won't allow storing the
            source-code or reliable reproduction of the runs.

        base_dir
            Optional full path to the base directory of this experiment. This
            will set the scope for automatic source file discovery.

        additional_host_info
            Optional dictionary containing as keys the names of the pieces of
            host info you want to collect, and as
            values the functions collecting those pieces of information.

        save_git_info:
            Optionally save the git commit hash and the git state
            (clean or dirty) for all source files. This requires the GitPython
            package.
        """

        caller_globals = inspect.stack()[1][0].f_globals
        if name is None:
            name = "trainer"

        ingredients = [datasets, models] + list(ingredients)
        Ingredient.__init__(
            self,
            path=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            _caller_globals=caller_globals,
            save_git_info=save_git_info,
        )
        self.datasets = datasets
        self.models = models
        self.get_datasets_list_command = None
        self.get_dataset_command = None
        self.get_dataset_iterator_command = None

        self.__validation_loaders = None
        self.__testing_loaders = None

        bind(self, self.command(plTrainer.__init__), "init_trainer")
        self.add_config(logger=CMD("/get_sacred_logger"))

        def get_trainer(self, *args, **kw):
            self.init_trainer(*args, **kw)
            return self

        # self.get_trainer = bind(self, self.command(get_trainer), "get_trainer")
        self.get_trainer = bind(self, get_trainer, "get_trainer")

        # self.command(get_dataset_iterator_command, unobserved=True)

    # @optional_kwargs_decorator
    # def trainer(self, function=None, prefix=None, unobserved=False, static_args={}, **extra_args):
    #     """
    #     Decorator to define a new Dataset.
    #
    #     The name of the dataset is used to get an instance of the dataset, it will register a command
    #
    #     Datasets are sacred commands.
    #
    #     The command can be given a prefix, to restrict its configuration space
    #     to a subtree. (see ``capture`` for more information)
    #
    #     A command can be made unobserved (i.e. ignoring all observers) by
    #     passing the unobserved=True keyword argument.
    #     :param function: the function to return a Dataset Object
    #     :param prefix: sacred configuration prefix
    #     :param unobserved: sacred unobserved
    #     :param static_args: static Args to be passed to the function, these arg need not to be serlizable and
    #      are not stored in the config
    #     :param extra_args: explicit arguments to be add to the config, you can these to override the function default
    #     values, for example wraping a config with CMD, then the parameter will be filled with excuting the command
    #     specified by CMD string value. CMD string have special context
    #     :return:
    #
    #
    #     """
    #     self.add_default_args_config(function, extra_args, static_args=static_args)
    #     captured_f = self.capture(function, prefix=prefix, static_args=static_args)
    #     captured_f.unobserved = unobserved
    #     self.commands[Trainer.TRAINER_STRING_PREFIX] = captured_f
    #     self.get_trainer = captured_f
    #     return captured_f
    def add_default_args_config(self, function, extra_args={}, static_args={}):
        """
        adds the default parameters of a function to the ingredient config at lowest priority!
        Default args config is meant remove the need to declare all the configurations manually.
        :param f: the function
        """
        # @todo get the doc of the params as well
        config_candidate = {**get_default_kwargs_dict(function), **extra_args}
        # remove "static_args" from config
        for k in static_args:
            config_candidate.pop(k, None)

        self.configurations.insert(0, self._create_config_dict(config_candidate, None))

    #
    # @optional_kwargs_decorator
    # def command(self, function=None, prefix=None, unobserved=False, static_args={}, **extra_args):
    #     """
    #     Decorator to define a new Dataset.
    #
    #     The name of the dataset is used to get an instance of the dataset, it will register a command
    #
    #     Datasets are sacred commands.
    #
    #     The command can be given a prefix, to restrict its configuration space
    #     to a subtree. (see ``capture`` for more information)
    #
    #     A command can be made unobserved (i.e. ignoring all observers) by
    #     passing the unobserved=True keyword argument.
    #     :param function: the function to return a Dataset Object
    #     :param prefix: sacred configuration prefix
    #     :param unobserved: sacred unobserved
    #     :param static_args: static Args to be passed to the function, these arg need not to be serlizable and
    #      are not stored in the config
    #     :param extra_args: explicit arguments to be add to the config, you can these to override the function default
    #     values, for example wraping a config with CMD, then the parameter will be filled with excuting the command
    #     specified by CMD string value. CMD string have special context
    #     :return:
    #
    #
    #     """
    #     self.add_default_args_config(function, extra_args, static_args=static_args)
    #     captured_f = self.capture(function, prefix=prefix, static_args=static_args)
    #     captured_f.unobserved = unobserved
    #     self.commands[function.__name__] = captured_f
    #     return captured_f

    def init_train_dataloader(self, model):
        """
        Dataloaders are provided by the model (adapted from lightening)
        :param model:
        :return:
        """
        training_sets = self.datasets.get_datasets(dict(train=True))
        if len(training_sets) == 0:
            raise MisconfigurationException(
                "You need to have at least one training set.\n with the config train=True."
                " run your script with '-p' to print the final config,"
                " training dataset should have the config  train=True."
            )
        if len(training_sets) > 1:
            warnings.warn(
                "Multiple training set is currently not supported, we will take the first one.\n\n"
                "Please report your use case\n\n "
            )
        self.get_train_dataloader = training_sets[0].get_iter

        # determine number of training batches
        if EXIST_ITER_DATASET and isinstance(
            self.get_train_dataloader().dataset, IterableDataset
        ):
            self.num_training_batches = float("inf")
        else:
            self.num_training_batches = len(self.get_train_dataloader())
            self.num_training_batches = int(
                self.num_training_batches * self.train_percent_check
            )

        # determine when to check validation
        # if int passed in, val checks that often
        # otherwise, it checks in [0, 1.0] % range of a training epoch
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
        else:
            self.val_check_batch = int(
                self.num_training_batches * self.val_check_interval
            )
            self.val_check_batch = max(1, self.val_check_batch)

        on_ddp = self.use_ddp or self.use_ddp2
        if on_ddp and not isinstance(
            self.get_train_dataloader().sampler, DistributedSampler
        ):
            msg = """
            You're using multiple gpus and multiple nodes without using a DistributedSampler
            to assign a subset of your data to each process. To silence this warning, pass a
            DistributedSampler to your DataLoader.

            ie: this:
            dataset = myDataset()
            dataloader = Dataloader(dataset)

            becomes:
            dataset = myDataset()
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = Dataloader(dataset, sampler=dist_sampler)

            If you want each process to load the full dataset, ignore this warning.
            """
            if msg not in self.shown_warnings and self.proc_rank == 0:
                self.shown_warnings.add(msg)
                warnings.warn(msg)

    def init_val_dataloader(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        # @todo add to documentation how to config a dataset for validation

        validation_sets = self.datasets.get_datasets(dict(validate=True))

        def get_validation_loaders():
            if self.__validation_loaders is None:
                self.__validation_loaders = [d.get_iter() for d in validation_sets]
                if len(self.__validation_loaders) == 0:
                    self.__validation_loaders = None
                    warnings.warn(
                        "no validation loaders available. this is not supported by the current trainer."
                    )
            return self.__validation_loaders

        self.get_val_dataloaders = get_validation_loaders

        # determine number of validation batches
        # val datasets could be none, 1 or 2+
        if self.get_val_dataloaders() is not None:
            self.num_val_batches = sum(
                len(dataloader) for dataloader in self.get_val_dataloaders()
            )
            self.num_val_batches = int(self.num_val_batches * self.val_percent_check)
            self.num_val_batches = max(1, self.num_val_batches)

        on_ddp = self.use_ddp or self.use_ddp2
        if on_ddp and self.get_val_dataloaders() is not None:
            for dataloader in self.get_val_dataloaders():
                if not isinstance(dataloader.sampler, DistributedSampler):
                    msg = """
                    Your val_dataloader(s) don't use DistributedSampler.

                    You're using multiple gpus and multiple nodes without using a
                    DistributedSampler to assign a subset of your data to each process.
                    To silence this warning, pass a DistributedSampler to your DataLoader.

                    ie: this:
                    dataset = myDataset()
                    dataloader = Dataloader(dataset)

                    becomes:
                    dataset = myDataset()
                    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                    dataloader = Dataloader(dataset, sampler=dist_sampler)

                    If you want each process to load the full dataset, ignore this warning.
                    """
                    if msg not in self.shown_warnings and self.proc_rank == 0:
                        self.shown_warnings.add(msg)
                        warnings.warn(msg)
                    break

    def init_test_dataloader(self, model):
        """Dataloaders are provided by the model.

        :param model:
        """

        testing_sets = self.datasets.get_datasets(dict(test=True))

        def get_testing_loaders():
            if self.__testing_loaders is None:
                self.__testing_loaders = [d.get_iter() for d in testing_sets]
            return self.__testing_loaders

        self.get_test_dataloaders = get_testing_loaders

        # determine number of test batches
        if self.get_test_dataloaders() is not None:
            len_sum = sum(len(dataloader) for dataloader in self.get_test_dataloaders())
            self.num_test_batches = len_sum
            self.num_test_batches = int(self.num_test_batches * self.test_percent_check)
            self.num_test_batches = max(1, self.num_test_batches)

        on_ddp = self.use_ddp or self.use_ddp2
        if on_ddp and self.get_test_dataloaders() is not None:
            for dataloader in self.get_test_dataloaders():
                if not isinstance(dataloader.sampler, DistributedSampler):
                    msg = """
                    Your `test_dataloader(s)` don't use DistributedSampler.

                    You're using multiple gpus and multiple nodes without using a
                    DistributedSampler to assign a subset of your data to each process.
                    To silence this warning, pass a DistributedSampler to your DataLoader.

                    ie: this::

                        dataset = myDataset()
                        dataloader = Dataloader(dataset)

                    becomes::

                        dataset = myDataset()
                        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                        dataloader = Dataloader(dataset, sampler=dist_sampler)

                    If you want each process to load the full dataset, ignore this warning.
                    """
                    if msg not in self.shown_warnings and self.proc_rank == 0:
                        self.shown_warnings.add(msg)
                        warnings.warn(msg)
                    break
