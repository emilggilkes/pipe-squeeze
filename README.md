PipeSqueeze
==============================

Compression experiments for pipeline parallel distributed deep learning

Run Instructions:
==============================
Experiments run use AWS Sagemaker and are launched from `src/launch_sagemaker.ipynb.` This requires you to have setup AWS Sagemaker yourself. You can do so following the quick start tutorial from AWS: https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html

In the notebook you will need to put in your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3 Bucket to write data to, and your Sagemaker ARN. Training is currently set up for nodes with at least 4 GPUs--the AmazonSagemaker instance used in the experiments is ml.g4dn.12xlarge. At least two instances are required to launch a training job with data parallelism. After that, you can simply run the notebook, changing any hyperparameters you wish.

References:
==============================
[PowerSGD](https://github.com/epfml/powersgd)

[GRACE](https://github.com/sands-lab/grace)

[On the Utility of Gradient Compression](https://github.com/uw-mad-dash/GradCompressionUtility)

PyTorch
