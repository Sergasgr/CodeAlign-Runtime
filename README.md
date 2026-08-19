# CodeAlign-Runtime

sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker

docker build -t codealign-runtime .
docker run --gpus all -it --rm --env-file .env -v $(pwd):/app codealign-runtime
python scripts/tu_script_baseline.py
