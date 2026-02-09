from routers.auth import router as auth_router
from routers.scenarios import router as scenarios_router
from routers.simulations import router as simulations_router
from routers.reflection import router as reflection_router
from routers.learning import router as learning_router
from routers.profile import router as profile_router

__all__ = [
    "auth_router",
    "scenarios_router",
    "simulations_router",
    "reflection_router",
    "learning_router",
    "profile_router"
]
