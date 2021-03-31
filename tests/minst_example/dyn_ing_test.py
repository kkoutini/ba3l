from sacred import Experiment as Sacred_Experiment, Ingredient
from sacred.config import DynamicIngredient
from sacred.config_helpers import CMD

print("TEST modifoed sacred for overridding experiments from child ")


# ing = Ingredient("other.sonofother")
# ing.add_config(a="sonofother value")
#
#
# ing = Ingredient("other",ingredients=[ing])
# ing.add_config(a="other value")
#


ex = Sacred_Experiment("top", ingredients=[])


@ex.config
def defa():
    test_command= CMD("/son.print_a")
    test_command2= CMD("/son2.print_a")


ex.add_config(
    son=DynamicIngredient(path="dyn_ing_test_son.ing", a="pkl", other="o"), o="o"
)
ex.add_config(son={"a": "top value"}, o="o")


@ex.named_config
def grand2():
    son = {"grand2": DynamicIngredient(path="dyn_ing_test_son.ing3", ok=1)}


@ex.named_config
def ing_someone():
    son = DynamicIngredient(path="dyn_ing_test_son.ing_someone", a="ing_someone")

@ex.named_config
def ing_son2():
    son2 = DynamicIngredient(path="dyn_ing_test_son.ing_someone", a="ing_someone")


@ex.command(prefix="son")
def print_a(a):
    print("a=", a)

@ex.command
def test(test_command):
    print(test_command)

@ex.command
def test2sons(test_command,test_command2):
    print(test_command)


@ex.named_config
def named():
    son = {"a": "top named"}
    other = "not important"


@ex.automain
def default_command():
    print(" Examples:\n"
          "python -m dyn_ing_test test with grand2   -p\n"
          "python -m dyn_ing_test test    -p\n"
          "python -m dyn_ing_test test2sons  with ing_son2  -p\n"
          "python -m dyn_ing_test test2sons  with ing_son2 ing_someone  -p\n"
          "python -m dyn_ing_test test with grand2  ing_someone -p\n")
    return print_a()



#  python -m dyn_ing_test test with grand2   -p
#  python -m dyn_ing_test test    -p

# python -m dyn_ing_test test with grand2  ing_someone -p
