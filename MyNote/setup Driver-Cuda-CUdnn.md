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









    




















