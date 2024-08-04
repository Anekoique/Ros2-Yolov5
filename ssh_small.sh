#!/bin/bash

gnome-terminal -t "ssh1" --window --geometry=85x25+50+50 -- bash -c "
ssh tds@192.168.1.101;
#source /opt/ros/humble/setup.bash;
cd ros2_ws;
source install/setup.sh;
exec bash;" 

gnome-terminal -t "image_pub" --window --geometry=85x25+830+50 -- bash -c "
ssh tds@192.168.1.101;
#source /opt/ros/humble/setup.bash;
cd ros2_ws;
./ros2_ws/image_pub.sh;
exec bash;"

gnome-terminal -t "ssh2"  --window --geometry=85x25+50+550 -- bash -c "
ssh tds@192.168.1.101;
#source /opt/ros/humble/setup.bash;
cd ros2_ws;
./mavros2.sh;
exec bash;"

gnome-terminal -t "image_sub"  --window --geometry=85x25+830+550 -- bash -c "
ros2 run topic2 image_sub;
exec bash;"

gnome-terminal -t "image_sub2"  --window --geometry=85x25+830+550 -- bash -c "
ros2 run topic2 show;
exec bash;"
