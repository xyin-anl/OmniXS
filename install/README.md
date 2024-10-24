```bash
docker build -t omnixas:bnl .
```

```bash
# docker run --gpus "device=0" -v /u/home/skharel/Downloads/OmniXAS:/workspace  -it --user $(id -u):$(id -g) omnixas:bnl bash
docker run --gpus all -v /u/home/skharel/Downloads/OmniXAS:/workspace  -it --user $(id -u):$(id -g) omnixas:bnl bash
python -m refactor.model.training --config-path ../../config/training --config-name expertXAS element=Cu type=FEFF name=expertXAS
```
