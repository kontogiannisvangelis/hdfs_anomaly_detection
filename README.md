```
DOCKER_BUILDKIT=1 sudo docker build -t hdfs_lad .
```

```
docker run -d --gpus all \
    -v $(pwd)/checkpoints/trainer:/checkpoints/trainer \
    -v $(pwd)/data:/data \
    hdfs_lad \
    /bin/bash -c "python3 /app/scripts/prepare_dataset.py && /app/scripts/grid_search.sh"
```

```

docker run -d --gpus all \
    -v $(pwd)/data:/data \
    hdfs_lad \
    /bin/bash -c "python3 /app/scripts/prepare_dataset.py"

```

```
rsync -avz \
    --exclude-from='.dockerignore' \
    -e "ssh -i ~/.ssh/key" \
    ./vkontogiannis@gpu.epu.ntua.gr:/mnt/datadrive/vkontogiannis \
    --dry-run
```
