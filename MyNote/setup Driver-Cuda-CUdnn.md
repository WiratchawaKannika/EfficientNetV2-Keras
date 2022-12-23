# Install NVIDIA Driver/Cuda toolkit/cudnn เครื่อง 28 สำหรับ NVIDIA GeForce RTX 3090Ti และ NVIDIA GeForce RTX 2080Ti

- [X] OS detail
  - Ubuntu 20.04/Linux x86_64

- [X] Graphic card detail (เดิม)

```
nvidia-smi -l

"NVIDIA-SMI 515.65.01
Driver Version: 515.65.01
CUDA Version: 11.7"
```

- [X] Finding out information about GPU

```
sudo lshw -C display
```

![lshw image](gisplay_lshw.png) 

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1. Install NVIDIA Driver
- [X] [Guide by NVIDIA](https://www.if-not-true-then-false.com/2021/debian-ubuntu-linux-mint-nvidia-guide/)

### 1.1 Check NVIDIA card supported

```
lspci |grep -E "VGA|3D"

"17:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A] (rev a1)
65:00.0 VGA compatible controller: NVIDIA Corporation Device 2203 (rev a1)"
```

### 1.2 Download NVIDIA Installer Package
Go to http://www.nvidia.com/Download/Find.aspx?lang=en-us 

- [X] GeForce RTX 30 series cards works with 520.xx, 515.xx, 510.xx, 470.xx NVIDIA drivers, (RTX 3090, RTX 3080, RTX 3070, RTX 3060 Ti, RTX 3060)
  - [ ] เราเลือก Version:	520.56.06
    
    ![driver image](nvidia_driver.png) 
    
```
wget https://in.download.nvidia.com/XFree86/Linux-x86_64/520.56.06/NVIDIA-Linux-x86_64-520.56.06.run
```
    
 ### 1.3 Make NVIDIA installer executable
 
 ```
chmod +x /path/to/NVIDIA-Linux-x86_64-520.56.06.run
 ```
  
 ### 1.4 Change root user
 
```
sudo su
```
 - ออกจาก root user
  
 ```
 CTRL+A+D
 ```
 
 ### 1.5 Make sure that system is up-to-date and running latest kernel, also make sure that don’t have any Debian / Ubuntu / Linux Mint / LMDE NVIDIA package installed
 
```
apt update
apt upgrade
apt autoremove $(dpkg -l xserver-xorg-video-nvidia* |grep ii |awk '{print $2}')
apt reinstall xserver-xorg-video-nouveau
```

- After update and/or NVIDIA drivers remove reboot system and boot using latest kernel and nouveau:

```
reboot
```

### 1.6 Install needed dependencies

```
apt install linux-headers-$(uname -r) gcc make acpid dkms libglvnd-core-dev libglvnd0 libglvnd-dev dracut
```

### 1.7 Disable nouveau

- 1.7.1 Create or edit /etc/modprobe.d/blacklist.conf

  - [ ] Append "blacklist nouveau"
 
 ```
echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf
```

- 1.7.2 Edit /etc/default/grub

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash rd.driver.blacklist=nouveau"
```

- 1.7.3 Update grub2 conf

```
update-grub2
```

- 1.7.4 Generate initramfs

```
## Backup old initramfs nouveau image ##
mv /boot/initrd.img-$(uname -r) /boot/initrd.img-$(uname -r)-nouveau
 
## Generate new initramfs image ##
dracut -q /boot/initrd.img-$(uname -r) $(uname -r)
```

### 1.8 Reboot to runlevel 3

```
systemctl set-default multi-user.target
reboot
```

### 1.9 Run NVIDIA Binary

```
sudo du
/path/to/NVIDIA-Linux-x86_64-520.56.06.run
```

- [X] NVIDIA Installer Installing Drivers
  
   ![confirm image](confirm_install.png) 
   
 ### 1.10 All Is Done and Then Reboot Back to Runlevel 5
 
 ```
systemctl set-default graphical.target
reboot
```
  - [X] succeed!!, Next step install "nvidia-smi":
  
 ```
sudo apt purge nvidia-*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-utils-520
sudo reboot
```

 - [X] Graphic card detail (ใหม่)

 ```
 nvidia-smi -l

 "NVIDIA-SMI 520.56.06
 Driver Version: 520.56.06
 CUDA Version: 11.8"
 ```

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2. Install NVIDIA CUDA Toolkit 11.8 

Go to https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local 

*Don't forget to log in before clicking this link.*

- [X] Installation Instructions:

  ```
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
  sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
  sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda
  ```
 
- [X] Checks CUDA toolkit is installed:

   ```
   systemctl status nvidia-persistenced
   ```

  From its [documentation](https://download.nvidia.com/XFree86/Linux-x86_64/396.51/README/nvidia-persistenced.html), nvidia-persistenced is intended to be run as a daemon from system initialization and is generally designed as a tool for compute-only      platforms where the NVIDIA device is not used to display a graphical user interface. If the daemon is not running, you can start/restart the daemon as follows
  
  ```
  sudo systemctl enable nvidia-persistenced
  ```
  
  - [X] To get the version of the NVIDIA driver, type
 
  ```
  cat /proc/driver/nvidia/version
  
  "NVRM version: NVIDIA UNIX x86_64 Kernel Module  520.61.05  Thu Sep 29 05:30:25 UTC 2022
  GCC version:  gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)"
  ```
  
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  ## 3. Install NVIDIA cuDNN
  
 In order to download cuDNN, ensure registered for the NVIDIA [Developer Program](https://developer.nvidia.com/developer-program).
 After logging in and accepting their terms and conditions, click on the following this links: [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
 
 ![chooseCUdnn image](chooseCUdnn.png) 
 
 - [X] Click Download. Then, ``scp cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb username@xx.xxx.xxx.xx:/home/username/downloaded``
 
    **NOTE** Absolutely do not download using the command ``wget`` เพราะจะทำให้ไฟล์เสียหาย และติดตั้งไม่ได้
 
 ### 3.1 Enable the local repository.
 
 ```
 sudo dpkg -i cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb
 ```
 
### 3.2 Import the CUDA GPG key.

```
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
```
หรือ หลังจาก run ``sudo dpkg -i cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb`` จะมี out put บอก command line ให้ใช้ Import the CUDA GPG key อยู่ด้วย

### 3.3 Refresh the repository metadata.

```
sudo apt-get update
```

### 3.4 Install library.

*ก่อนทำขั้นตอนนี้ ให้เข้าไป check library ใน ``cd /var/cudnn-local-repo-ubuntu2004-8.5.0.96`` ควรจะมี 3 ไฟล์ ดังนี้*

```
- libcudnn8_8.5.0.96-1+cuda11.7_amd64.deb
- libcudnn8-dev_8.5.0.96-1+cuda11.7_amd64.deb
- libcudnn8-samples_8.5.0.96-1+cuda11.7_amd64.deb
```

#### 3.4.1 Install the runtime library

```
sudo apt-get install libcudnn8=8.5.0.96-1+cuda11.7
```

#### 3.4.2 Install the developer library.

```
sudo apt-get install libcudnn8-dev=8.5.0.96-1+cuda11.7
```
 
#### 3.4.3 Install the code samples and the cuDNN library documentation.

```
sudo apt-get install libcudnn8-samples=8.5.0.96-1+cuda11.7
```

### 3.5 Verifying the Install on Linux

To verify that cuDNN is installed and is running properly, compile the mnistCUDNN sample located in the /usr/src/cudnn_samples_v8 directory in the Debian file. 

*ก่อนทำขั้นตอนนี้เข้าไป Check ให้แน่ใจก่อนว่ามี Folder ``mnistCUDNN`` ใน ``cd /usr/src/cudnn_samples_v8/cudnn_samples_v8``*

```
sudo reboot
```

#### 3.5.1 Copy the cuDNN samples to a writable path.

```
cp -r /usr/src/cudnn_samples_v8/ $HOME
```

#### 3.5.2 Go to the writable path.

```
cd  $HOME/cudnn_samples_v8/mnistCUDNN
```

#### 3.5.3 Compile the mnistCUDNN sample

```
./mnistCUDNN
```

- [X] If cuDNN is properly installed and running on Linux system, will see a message similar to the following:

  ```
  Test passed!
  ```
  
  ![test_pass image](test_pass.png) 
  
- [X] PS. ถ้า test แล้วมี ``fatal error: FreeImage.h``

  ```
  test.c:1:10: fatal error: FreeImage.h: No such file or directory
      1 | [[include]] "FreeImage.h"
  ```
  
  ![Free_Image](FreeImage.png) 
  
  - [ ] Solution: run command line
  
    ```
    sudo apt-get install libfreeimage3 libfreeimage-dev
    ```
  
    Then, run the mnistCUDNN sample again!!
  
     ```
    make clean && make
    ./mnistCUDNN
    ```
  
### 3.6 Check NVIDIA cudnn version

```
dpkg -l | grep cudnn

"ii  cudnn-local-repo-ubuntu2004-8.5.0.96       1.0-1                               amd64        cudnn-local repository configuration files
ii  libcudnn8                                  8.5.0.96-1+cuda11.7                 amd64        cuDNN runtime libraries
ii  libcudnn8-dev                              8.5.0.96-1+cuda11.7                 amd64        cuDNN development libraries and headers
ii  libcudnn8-samples                          8.5.0.96-1+cuda11.7                 amd64        cuDNN samples"
```

```
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

```

```
nvidia-smi
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

  
  


  
  



    




















