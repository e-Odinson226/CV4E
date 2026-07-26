---
date: 2026-05-29
---
# Todos:

- [x] check vjepa’s evaluation pipelines
- [x] load the EpicKitchen dataset
- [x] run the pipeline with the EpicKitchen dataset

  

---

# Notes

**Inference from existing probes**

Use provided inference configs under [Evaluation Attentive Probes](https://github.com/facebookresearch/vjepa2#evaluation-attentive-probes). Download the corresponding checkpoint, rename it to 'latest.pt', and create a folder with the checkpoint inside, with the format matching the variables in the config:

```
[folder]/[eval_name]/[tag]/latest.pt
```

Then run inference, locally or distributed, using the same evaluation commands as above, but with configs from `configs/inference`.

**Pretraining**

Likewise, training can also be run locally or distributed. Pretraining and cooldown training phases are run with the same command using different configs. These sample commands launch initial training of a ViT-L model. Configs for cooldown (or action-conditioned) training can be found in the same directory as the config for initial training.

**Local**

```
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0
```

**Distributed**

```
python -m app.main_distributed \
  --fname configs/train/vitl16/pretrain-256px-16f.yaml
  --time 6000
  --account my_account --qos=my_qos
```

**Postraining**

Post-training of the action-conditioned model, starting from the pretrained VJEPA 2 backbone, also follows a similar interface, and can be run locally or distributed using [this config](https://github.com/facebookresearch/vjepa2/blob/main/configs/train/vitg16/droid-256px-8f.yaml). We post-train the model starting from the ViT-g/16 backbone.

**Local**

```
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml \
  --devices cuda:0
```

**Distributed**

```
python -m app.main_distributed \
  --fname configs/train/vitg16/droid-256px-8f.yaml
  --time 6000
  --account my_account --qos=my_qos
```

---

## Load EpicKitchen-100

1. get the torrent file
2. extract the ids in the torrent

```
python -c "
from torrentool.api import Torrent
torrent = Torrent.from_file('/mnt/data/home/zj2433/Projects/Ego/data/epic-kitchen/original-ek100.torrent')
for idx, f in enumerate(torrent.files, start=1):
    print(f'{idx}|{f.name}')
" > complete_file_list.txt
```

1. modify the desired ids

```
grep "P01/" complete_file_list.txt | cut -d'|' -f1 | paste -sd, - > p01_indices.txt
```

1. download

```
aria2c --select-file=`cat p01_indices.txt` --seed-time=0 --dir=/mnt/data/home/zj2433/Projects/Ego/data/ek100/videos --max-connection-per-server=4 --split=4 --file-allocation=none /mnt/data/home/zj2433/Projects/Ego/data/epic-kitchen/ek100.torrent
```