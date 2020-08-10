tag=${1}
reg=${2}
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${reg}.dkr.ecr.us-east-1.amazonaws.com
docker build -t multimodal-training -f training-container/Dockerfile.training .
docker tag multimodal-training:latest ${reg}.dkr.ecr.us-east-1.amazonaws.com/multimodal-training:${tag}
docker push ${reg}.dkr.ecr.us-east-1.amazonaws.com/multimodal-training:${tag}
docker rmi multimodal-training
docker image prune -f
