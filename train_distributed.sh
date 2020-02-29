NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/small_obs/small_obs.yaml
