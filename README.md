```
sudo docker build -t hdfs_anomaly_detection_env .
```

```
sudo docker run -d  --gpus all -v $(pwd):/app hdfs_anomaly_detection_env /bin/bash -c "python3 /app/scripts/prepare_dataset.py && /app/scripts/grid_search.sh"

```