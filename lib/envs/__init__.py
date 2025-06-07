from lib.envs.garcia_cicco_et_al_2010 import GarciaCiccoEnv
from .ramsey import RamseyEnv

NAME_TO_ENV = {
    "Ramsey": RamseyEnv,
    "GarciaCicco_et_al_2010": GarciaCiccoEnv,
}
