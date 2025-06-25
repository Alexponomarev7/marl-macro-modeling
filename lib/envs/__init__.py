from lib.envs.garcia_cicco_et_al_2010 import GarciaCiccoEnv
from lib.envs.mccandless_2008_9 import McCandless2008Ch9Env
from .ramsey import RamseyEnv

NAME_TO_ENV = {
    "Ramsey": RamseyEnv,
    "GarciaCicco_et_al_2010": GarciaCiccoEnv,
    "McCandless_2008_Chapter_9": McCandless2008Ch9Env,
}
