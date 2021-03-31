
# Ba3l: the sacred lightening of a torch!

**The researcher friendly pytorch environment. Ba3l= sacred+pytorch-lightening**



## Docs   
**[View the docs here](https:///)**


## Why Ba3l? 
The main design principle is to be ETHIC (Every THing Is a Config).
Sacred provides a very research friendly tools for configuration and logging experiments.
Lightening for all the features that you'd expect from the pytorch wraper!
We redesign much of lightening Object-Oriented desgin to fit with ETHIC.
The result is scalable, very modular, very configuarble framework.

## Caution
Sacred and pytorch-lightening should not be installed on the same environment as Ba3l, since there is a modified and packaged version with pyzeus.

If the Sacred and pytorch-lightening versions were not installed automaticaly. you can use the following commands:
```shell script
pip install https://github.com/kkoutini/sacred/archive/ba3l.zip
# or
pip install https://github.com/kkoutini/pytorch-lightning/archive/ba3l.zip
```

### Sacred modifications
- Allow adding configs dynamically from command line.
- Allow Ingredients to override configuration of there sub-ingredients. For example, you can override any dataset configurations from the experiment configs.
- Add feature to automaticaly capture a function or a class and add its arguments as config. This allows to use the default values of a function as a sacred config, instead of explicitly adding them.
- Defining a new type of config `CMD` which allows to specify a function by it's name in config. Examples `iter` or `dataset`.

### Pytorch-lightening modifications
- Allow multiple inhertince to use pytorch-lightening trainer as sacred ingredient.

## Features
- We Kept all the features of [sacred](https://github.com/IDSIA/sacred) and [pytorch-lightening](pytorch-lightning).
- We use sacred to capture the workflow of lightening which allows to modify the behaviour of lightening using sacred configs (default configs, named configs, command-line configs, etc... refer to [sacred docs](https://sacred.readthedocs.io/en/stable/)).
- All the loggers supported by both lightening and Sacred (WIP), It's recommended to use sacred for meta analysis for a big chunk of experiments and tensorboard (for example) for specific experiment.
- We introduced CMD config, a special string that refers to a function (sacred command). This allows to seamlessly using functions in the config. and allows further flexibility to modify the behaviour using only config.
- We allow capturing of any function and class, and adding it's arguments to the config (with the default values). You don't have to explicitly declare the config params. For example: capturing the Dataloader will add batch_size, shuffle, etc.. to the config and pass them automatic when the function is called.
for a brief overview take a look at the tests/examples.
 
## Examples


 
