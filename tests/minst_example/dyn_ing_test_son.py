from sacred import Experiment as Sacred_Experiment, Ingredient
from sacred.config import DynamicIngredient

print("TEST modifoed sacred for overridding experiments from child, son import ")

ing = Ingredient("son")
ing.add_config(a="son value", son_other_config="son_other_config value")

ing2 = Ingredient("dynamic_grand_son")

ing2.add_config(a="grand son value", gson_other_config="grand son_other_config value")

ing3 = Ingredient("dynamic_grand_son2")

ing3.add_config(
    a="grand son 22 value", gson_other_config="grand son_other_config  22 value"
)


ing.add_config(sonofson=DynamicIngredient("dyn_ing_test_son.ing2"))


ing_someone = Ingredient("son")
ing_someone.add_config(a="ing_someone value", ing_someone="son_other_config value", ot="son ot")


@ing.command
def print_a(a):
    print("ing a=", a)

@ing_someone.command
def print_a(a):
    print("ing_someone a=", a)
