#!/bin/bash
set -euo pipefail

# run this script using `sudo -E ./bootstrap.sh`, otherwise it won't work
################################################################################

# external vars the user may override
################################################################################
WITH_AUTOUPDATES=${WITH_AUTOUPDATES:-1}
NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION:-535}
NO_LAUNCH=${NO_LAUNCH:-0}


# # internal vars
# ################################################################################
REBOOT_REQUIRED=0
DEBIAN_FRONTEND=noninteractive
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

function echo_() {
  echo "# $@"
  echo "################################################################################"
}


# check for root and setup exit trap
################################################################################
if [[ $(id -u) -ne 0 ]]; then
  echo_ "Please run this script as root."
  exit 1
fi

function on_exit_ {
  echo_ cleaning up...
  apt-mark unhold openssh-server
  systemctl enable unattended-upgrades
  systemctl start unattended-upgrades
}
trap on_exit_ INT TERM EXIT

# Stop unattended upgrades
################################################################################
systemctl stop unattended-upgrades
systemctl disable unattended-upgrades


# setup base files/folders
################################################################################
echo_ setting up base files and folders
touch $HOME/.bashrc
chown $SUDO_USER:$SUDO_USER $HOME/.bashrc
chmod 644 $HOME/.bashrc

mkdir -p $HOME/.local/bin
if ! [[ $(echo $PATH | grep "$HOME/.local/bin") ]]; then
  # add ~/.local/bin to the path
  export PATH="$HOME/.local/bin:$PATH "

  # and ensure it's there in future
  echo "" >> $HOME/.bashrc
  echo "export PATH=$HOME/.local/bin:$PATH" >> $HOME/.bashrc
fi

chown -R $SUDO_USER:$SUDO_USER $HOME/.local

# nvidia drivers
################################################################################
echo_ checking for nvidia drivers
if ! [[ $(which nvidia-smi) ]]; then
  echo_ nvidia drivers were not found, installing...

  apt purge nvidia-*
  add-apt-repository -y ppa:graphics-drivers/ppa
  apt update -qq
  apt-mark unhold nvidia* libnvidia*
  apt install -y libnvidia-common-$NVIDIA_DRIVER_VERSION libnvidia-gl-$NVIDIA_DRIVER_VERSION nvidia-driver-$NVIDIA_DRIVER_VERSION
  dpkg-query -W --showformat='${Package} ${Status}\n' | grep -v deinstall | awk '{ print $1 }' | grep -E 'nvidia.*-[0-9]+$' | xargs -r -L 1 apt-mark hold

  # signal to the script that it _should_ reboot the server once done.
  REBOOT_REQUIRED=1
fi


# nvidia/cuda/container-toolkit
################################################################################
echo_ checking for the nvidia container toolkit
if ! [[ $(command nvidia-ctk) ]]; then
  echo_ the nvidia container toolkit was not found, installing...
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  rm cuda-keyring_1.1-1_all.deb

  apt update -qq
  apt-mark unhold nvidia* libnvidia*
  apt install -y nvidia-container-toolkit
  dpkg-query -W --showformat='${Package} ${Status}\n' | grep -v deinstall | awk '{ print $1 }' | grep -E 'nvidia.*-[0-9]+$' | xargs -r -L 1 apt-mark hold
fi

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker


# do not upgrade openssh server whilst installing
################################################################################
apt-mark hold openssh-server

# fix anything broken, update stuff, and install base software
################################################################################
echo_ setting up base packages
apt update -qq
apt install -y vim git curl wget cron net-tools dnsutils software-properties-common


# python 3.10
################################################################################
if ! command -v python &> /dev/null; then
    echo_ "Python not found, installing Python 3.10..."
    apt update
    apt install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt update
    apt install -y python3.10 python3.10-venv python3.10-distutils
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
    update-alternatives --set python /usr/bin/python3.10

    # Install pip for Python 3.10
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    rm get-pip.py

    echo_ "Python 3.10 installed successfully"
else
    echo_ "Python is already installed"
fi


VENV_PATH="$HOME/.venv"
if [ ! -d "$VENV_PATH" ]; then
    python -m venv $VENV_PATH
    echo ". $VENV_PATH/bin/activate" >> $HOME/.bashrc
    chown -R $SUDO_USER:$SUDO_USER $VENV_PATH $HOME/.bashrc
    echo_ "Python venv created"
    . $VENV_PATH/bin/activate
else
    echo_ "Python venv already exists at $VENV_PATH"
fi

# docker
################################################################################
echo_ "checking for docker"
if ! [[ $(which docker) ]]; then
  echo_ docker was not found, installing...
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io
  systemctl enable --now docker
fi

echo_ "checking for docker-compose"
if ! [[ $(which docker-compose) ]]; then
  echo_ "docker-compose was not found, installing..."
  apt-get update
  apt-get install -y docker-compose-plugin
fi

groupadd docker || true
usermod -aG docker $SUDO_USER || true

# pm2 & jq
################################################################################
echo_ "checking for pm2 & jq"
apt-get install -y -qq nodejs npm
npm i -g -q pm2
apt-get install -y -qq jq


# Nano for config
################################################################################
echo_ "checking for nano"
if ! [[ $(which nano) ]]; then
  echo_ "nano was not found, installing..."
  apt-get install -y -qq nano
fi

# Task for taskfile
################################################################################
echo_ "checking for task"
if ! [[ $(which task) ]]; then
  echo_ "task was not found, installing..."
  sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
  export PATH="$HOME/.local/bin:$PATH"
  echo_ "task installed successfully"
fi

# configure servers to start on boot
################################################################################
if [[ NO_LAUNCH -eq 1 ]]; then
  :
else
  if [[ WITH_AUTOUPDATES -eq 1 ]]; then
    . $HOME/.venv/bin/activate
  else
    docker compose --env-file .vali.env -f ops/compose/docker-compose.yml up -d --build
  fi
fi


# finally, reboot if needed
################################################################################
if [[ REBOOT_REQUIRED -eq 1 ]]; then
  echo_ "bootstrap.sh modified something that requires a reboot. Please SSH back in to this machine after a short while :)"
  shutdown now -r
else
  echo_ "bootstrap.sh is all done :)"
fi

echo_ "Please run the following command!!"
echo_ ". ~/.bashrc"
