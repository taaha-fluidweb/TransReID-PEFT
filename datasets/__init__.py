def __getattr__(name):
    if name == "make_dataloader":
        from .make_dataloader import make_dataloader
        return make_dataloader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")