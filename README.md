# Deploy Multiple Object Tracking and Pose Estimation YoloV8 Model with Amazon SageMaker


[Multiple Object Tracking](https://motchallenge.net/) or MOT estimates a bounding box and ID for each pre-defined object in videos or consecutive frames. [Pose Estimation](https://en.wikipedia.org/wiki/Pose_(computer_vision)) estimate keypoints on human body These two tasks, has been used in live sports, manufacturing, surveillance, and traffic monitoring. In the past, the high latency caused by the limitation of hardware and complexity of ML-based tracking algorithm is a major obstacle for its application in the industry.

This post shows how to deploy a pretrained [YoloV8](https://docs.ultralytics.com/) model with Amazon SageMaker local mode and real-time inference endpoint.

## Prerequisites
- Create an AWS account or use the existing AWS account.
- This notebook can run on CPU or GPU instances, the default instances used are **m5.xlarge** EC2 instance and **ml.m5.2xlarge** SageMaker Endpoint instance.
- This notebook is designed to run on SageMaker Studio Domain with **VPCOnly** mode. Check [On Boarding SageMaker Studio with VPC](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-vpc.html) for more information.
- For IAM role, choose the existing IAM role or create a new IAM role, attach the policy of AmazonSageMakerFullAccess and AmazonElasticContainerRegistryPublicFullAccess to the chosen IAM role. 
- If using SageMaker Studio to run this notebook, make sure prerequisites for [SageMaker Studio Docker CLI extension](https://github.com/aws-samples/sagemaker-studio-docker-cli-extension#prerequsites) are also satisfied.

## Serving

We provide two ways of deploying the pretrained model: local mode endpoint and real time inference endpoint on SageMaker.
- To deploy the endpoints, open [`inference-YoloV8.ipynb`](inference-YoloV8.ipynb) and run the cells step by step.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
