from types import SimpleNamespace

C = SimpleNamespace(
    ENV_ID = "Acrobot-v1",

    #Jack
    HIDDEN = 128,
    GAMMA = 0.99,
    TAU = 0.005,
    GRAD_CLIP = 10.0,
    DOUBLE = False,

    #GOM
    LR = 1e-3,

    #Oisin
)