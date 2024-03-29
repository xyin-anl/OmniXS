import hydra

cfg = None
if not hydra.core.global_hydra.GlobalHydra().is_initialized():
    print(hydra.core.global_hydra.GlobalHydra())
    with hydra.initialize(version_base=None, config_path="."):
        cfg = hydra.compose(config_name="defaults")


if __name__ == "__main__":
    # print(cfg.paths.poscar.format(compound="Cu", id="mp-1478"))

    @hydra.main(version_base=None, config_path=".", config_name="defaults")
    def main(cfg):
        model_name = "ft_tl"
        cfg = hydra.compose(
            config_name="defaults", overrides=[f"model_name={model_name}"]
        )
        model = hydra.utils.instantiate(cfg.model)
        print(model)

    main()
