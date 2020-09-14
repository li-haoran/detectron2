
# PointsCollection in Detectron2
**A Foundation for Instance Segmentation**


<div align="center">
  <img src="https://raw.githubusercontent.com/li-haoran/detectron2/master/projects/PointsColletion/test.png" width="600px" />
</div>

In this repository, we release code for PointsCollection in Detectron2.
PointsCollection is a points-based instance segmentation project, which collected all the supporting points for object semantic,
 and reconstruct the instance segmentation mask. 

## Installation
First install Detectron2 following the [documentation](https://detectron2.readthedocs.io/tutorials/install.html) and
[setup the dataset](../../datasets). Then compile the PointsCollection-specific op (`points_collection_ops`):
```bash
cd /path/to/detectron2/projects/Pointscollection/layers/points_collection_ops
python setup.py build develop
mv build/lib/xx.so xx.so
```

## Training

To train a model, run:
```bash
python /path/to/detectron2/projects/PointsCollection/train_net.py --config-file <config.yaml>
```

For example, to launch PointsCollection simple bottom-up training (1x schedule) with ResNet-50 backbone on 4 GPUs,
one should execute:
```bash
python /path/to/detectron2/projects/PointsCollection/train_net.py --config-file configs/PointsCollection_R_50_1x.yaml --num-gpus 4
```

## Evaluation

Model evaluation can be done similarly (1x schedule with scale augmentation):
```bash
python /path/to/detectron2/projects/PointsCollection/train_net.py --config-file configs/PointsCollection_R_50_1x.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

# Pretrained Models

| Backbone | lr sched | AP box | AP mask | download                                                                                                                                    |
| -------- | -------- | --     | ---  | --------                                                                                                                                    |
| R50      | 1x       | xx.x   | xx.x | ....|

## <a name="CitingPointsCollection"></a>Citing PointsCollection

If you use PointsCollection, please use the following BibTeX entry.

```
xxx
```

