tag=${1}
reg=${2}
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${reg}.dkr.ecr.us-east-1.amazonaws.com
docker build -t multimodal-inference -f inference-container/Dockerfile.inference .
docker tag multimodal-inference:latest ${reg}.dkr.ecr.us-east-1.amazonaws.com/multimodal-inference:${tag}
docker push ${reg}.dkr.ecr.us-east-1.amazonaws.com/multimodal-inference:${tag}
docker rmi multimodal-inference
docker image prune -f
