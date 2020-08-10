tag=${1}
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 651826702434.dkr.ecr.us-east-1.amazonaws.com
docker build -t multimodal-training -f training-container/Dockerfile.training .
docker tag multimodal-training:latest 651826702434.dkr.ecr.us-east-1.amazonaws.com/multimodal-training:${tag}
docker push 651826702434.dkr.ecr.us-east-1.amazonaws.com/multimodal-training:${tag}
docker rmi multimodal-training
docker image prune -f
