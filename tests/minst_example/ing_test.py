from sacred import Experiment as Sacred_Experiment, Ingredient

print("TEST modifoed sacred for overridding experiments from child ")

ing = Ingredient("son")
ing.add_config(a="son value")


@ing.command
def print_a(a):
    print("a=", a)


ex = Sacred_Experiment("top", ingredients=[ing])

ex.add_config(son={"a": "top value"}, o="o")


@ex.named_config
def named():
    son = {"a": "top named"}
    other = "not important"


@ex.automain
def default_command():
    return print_a()
