# One image for training (GPU), sim demo collection and sim evaluation (headless EGL rendering).
# Build from the DeviceAssembly directory (it copies both policy/ and sim/):
#   docker buildx build --platform linux/amd64 -f policy/docker/Dockerfile -t dit-policy:latest .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    TORCH_HOME=/opt/torch_cache
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libgles2 libglvnd0 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY teleop/policy/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Compute nodes may have no internet: bake the ImageNet ResNet-18 weights into the image.
RUN python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')" && \
    chmod -R a+rX /opt/torch_cache

WORKDIR /opt/device_assembly
COPY teleop/policy/dit_policy policy/dit_policy
COPY teleop/policy/configs policy/configs
COPY sim/coral_sim sim/coral_sim
COPY sim/config sim/config
COPY sim/tools sim/tools
COPY sim/assets/menagerie/universal_robots_ur5e sim/assets/menagerie/universal_robots_ur5e
COPY sim/assets/menagerie/robotiq_2f85 sim/assets/menagerie/robotiq_2f85
COPY sim/assets/meshes/visual/*_vis.stl sim/assets/meshes/visual/
COPY sim/assets/meshes/collision sim/assets/meshes/collision
ENV PYTHONPATH=/opt/device_assembly/policy:/opt/device_assembly/sim
