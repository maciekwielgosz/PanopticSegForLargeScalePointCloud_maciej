#!/bin/bash

CONTAINER_NAME="test_e2e_instance"
IMAGE_NAME="nibio/e2e-instance"

# Check if the container exists
if [ $(docker container ls -a -q -f name=$CONTAINER_NAME) ]; then
    echo "Removing existing container $CONTAINER_NAME"
    docker container rm $CONTAINER_NAME
else
    echo "Container $CONTAINER_NAME does not exist."
fi

# Check if the image exists
# if [ $(docker image ls -q -f reference=$IMAGE_NAME) ]; then
#     echo "Removing existing image $IMAGE_NAME"
#     docker image rm $IMAGE_NAME
# else
#     echo "Image $IMAGE_NAME does not exist."
# fi

./build.sh

echo "Running the container"
# docker run -it --gpus all --name $CONTAINER_NAME $IMAGE_NAME > e2e-instance.log 2>&1


# for local testing
epochs=3
batch_size=8
cuda=0 # -1 for cpu, 0 for gpu

docker run -it --gpus all \
    -e epochs=$epochs \
    -e batch_size=$batch_size \
    -e cuda=$cuda \
    --name $CONTAINER_NAME \
    --mount type=bind,source=/home/nibio/mutable-outside-world/code/PanopticSegForLargeScalePointCloud_maciej/data_bucket,target=/home/data_bucket \
    $IMAGE_NAME




