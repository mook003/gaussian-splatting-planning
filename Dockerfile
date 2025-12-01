# Use ROS 2 Humble from Docker Hub as the base image
FROM ros:humble-ros-base

# Set ROS distro
ENV ROS_DISTRO=humble

# Set arguments for user creation
ARG USERNAME=mobile
ARG USER_UID=1000
ARG USER_GID=$USER_UID

#Create USER
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Update and install necessary packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y sudo curl gnupg2 lsb-release net-tools python3-pip \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list

# Обновление и установка базовых пакетов
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    sudo \
    git \
    wget \
    nano \
    python3-pip \
    ros-${ROS_DISTRO}-tf2-tools \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-hardware-interface \
    ros-${ROS_DISTRO}-urdf \
    ros-${ROS_DISTRO}-urdfdom \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-rviz-default-plugins \
    ros-${ROS_DISTRO}-rqt-robot-steering \
    ros-${ROS_DISTRO}-rqt-tf-tree \
    ros-${ROS_DISTRO}-nav2-rviz-plugins \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-pointcloud-to-laserscan \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-image-transport-plugins \ 
    ros-${ROS_DISTRO}-compressed-image-transport \
    ros-${ROS_DISTRO}-image-publisher \
    ros-${ROS_DISTRO}-camera-info-manager \
    ros-${ROS_DISTRO}-diagnostic-updater \ 
    ros-${ROS_DISTRO}-diagnostic-msgs \
    ros-${ROS_DISTRO}-statistics-msgs \
    ros-${ROS_DISTRO}-backward-ros \
    ros-${ROS_DISTRO}-camera-calibration-parsers \
    ros-${ROS_DISTRO}-image-publisher \
    ros-${ROS_DISTRO}-teleop-twist-keyboard \
    ros-${ROS_DISTRO}-imu-tools \
    libgflags-dev \
    libdw-dev \
    nlohmann-json3-dev  \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    at-spi2-core \
    x11-apps \
    xauth \
    python3-colcon-common-extensions \
    --fix-missing

RUN pip install \
    pyserial \
    spidev \ 
    open3d \
    "numpy<2" --upgrade

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Initialize rosdep (run as user)
RUN sudo rosdep init || true \
    && rosdep update

# Добавляем source в .bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/$USERNAME/.bashrc

USER $USERNAME

RUN mkdir -p /home/$USERNAME/ros2_ws/src

CMD ["bash"]
