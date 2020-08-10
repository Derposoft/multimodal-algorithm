tag=${1}
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 651826702434.dkr.ecr.us-east-1.amazonaws.com
docker build -t multimodal-inference -f inference-container/Dockerfile.inference .
docker tag multimodal-inference:latest 651826702434.dkr.ecr.us-east-1.amazonaws.com/multimodal-inference:${tag}
docker push 651826702434.dkr.ecr.us-east-1.amazonaws.com/multimodal-inference:${tag}
docker rmi multimodal-inference
docker image prune -f
