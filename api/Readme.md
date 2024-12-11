


`sudo bash ./install.sh`

to redo all the steps remove the lock files

`rm ${ROOT}/opt/swarms/install/* `

or in my system:
```
export ROOT=/mnt/data1/swarms
sudo rm ${ROOT}/opt/swarms/install/*
```

rerun
```
export ROOT=/mnt/data1/swarms; 
sudo rm ${ROOT}/opt/swarms/install/*; 
sudo bash ./install.sh
```
