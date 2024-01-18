import hydra

cfg = None
if not hydra.core.global_hydra.GlobalHydra().is_initialized():
    print(hydra.core.global_hydra.GlobalHydra())
    with hydra.initialize(version_base=None, config_path="."):
        cfg = hydra.compose(config_name="defaults")


if __name__ == "__main__":
    print(cfg.paths.poscar.format(compound="Cu", id="mp-1478"))
