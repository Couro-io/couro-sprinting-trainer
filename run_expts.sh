# Run pose-estimation using yolov7
#python3 ./couro/experiments/01_pose_estimation/inference.py

# Run action-recognition using ip-CSN-152
docker run -v ${PWD}:/app -w /app mmaction2:latest python3 ./couro/experiments/02_sports_recognition/inference.py > ./couro/experiments/02_sports_recognition/results.json