_base_ = ["./semseg-pt-v3m1-0-base.py"]

# Fast end-to-end smoke test: 1 epoch, few iterations, then a val pass.
epoch = 1
eval_epoch = 1
batch_size = 4

data = dict(
    train=dict(loop=2),  # 3 scenes x 2 = 6 samples -> a couple of iterations
)
