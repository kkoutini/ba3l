from importlib import import_module

from ba3l.ingredients.datasets import Datasets
from ba3l.ingredients.models import Models, Model
from ba3l.trainer import Trainer
from ba3l.util.sacred_logger import SacredLogger
from sacred import Experiment as Sacred_Experiment, Ingredient
from typing import Sequence, Optional, List

from sacred.commandline_options import CLIOption
from sacred.host_info import HostInfoGetter
from sacred.utils import PathType
from pytorch_lightning import loggers as pl_loggers


def ingredients_recursive_apply(ing, fn):
    fn(ing)
    for kid in ing.ingredients:
        ingredients_recursive_apply(kid, fn)


def get_loggers(expr, use_tensorboard_logger=False):
    sacred_logger = SacredLogger(expr)
    loggers = [sacred_logger]
    if use_tensorboard_logger:
        loggers.append(pl_loggers.TensorBoardLogger(sacred_logger.name))
    return loggers

class Experiment(Sacred_Experiment):
    """
    Main Ba3l Experiment class overrides sacred experiments.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        ingredients: Sequence[Ingredient] = (),
        datasets: Optional[Ingredient] = None,
        trainer: Optional[Ingredient] = None,
        models: Optional[Ingredient] = None,
        interactive: bool = False,
        base_dir: Optional[PathType] = None,
        additional_host_info: Optional[List[HostInfoGetter]] = None,
        additional_cli_options: Optional[Sequence[CLIOption]] = None,
        save_git_info: bool = True,
    ):
        """
        Create a new experiment with the given name and optional ingredients. (from Sacred)


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
        if models is None:
            models = Models.get_instance()
        self.models = models
        if datasets is None:
            datasets = Datasets.get_instance()
        self.datasets = datasets
        if trainer is None:
            trainer = Trainer.get_instance(datasets=datasets, models=models)
        self.trainer = trainer
        if ingredients is None:
            ingredients = []
        ingredients = list(ingredients) + [models, datasets, trainer]
        super().__init__(
            name=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            additional_host_info=additional_host_info,
            additional_cli_options=additional_cli_options,
            save_git_info=save_git_info,
        )
        self.trainer.command(get_loggers, static_args={"expr": self})
        # filling out Default config

    def get_trainer(self, *args, **kw):
        return self.trainer.get_trainer(*args, **kw)

    def get_dataloaders(self, filter={}):
        results = {}
        for ds in self.datasets.get_datasets(filter):
            results[ds.name] = ds.get_iterator()
        if len(results) == 1:
            for k, v in results.items():
                return v
        return results

    def get_train_dataloaders(self):
        return self.get_dataloaders(dict(train=True))

    def get_val_dataloaders(self):
        return self.get_dataloaders(dict(validate=True))

    def _create_run(
        self,
        command_name=None,
        config_updates=None,
        named_configs=(),
        info=None,
        meta_info=None,
        options=None,
        dry_run=False,
    ):
        if self.current_run is not None:
            # @todo replace with logger
            print("Warning: multiple runs are not yet supported")

        run = super()._create_run(
            command_name,
            config_updates,
            named_configs,
            info,
            meta_info,
            options,
            dry_run=True,
        )

        # lazy model loading
        for k, v in run.config.get("models", {}).items():
            cls = getattr(
                import_module(v["path"].rsplit(".", 1)[0]), v["path"].rsplit(".", 1)[1]
            )
            model = Model("models." + k)
            model.instance(cls)
            self.models.ingredients.append(model)
        run = super()._create_run(
            command_name,
            config_updates,
            named_configs,
            info,
            meta_info,
            options,
            dry_run=False,
        )

        def update_current_run(ing):
            ing.current_run = run

        ingredients_recursive_apply(self, update_current_run)

        return run
