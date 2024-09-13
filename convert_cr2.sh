#!
for pic in cr2/*.CR2
do
     darktable-cli "$pic" "$(basename ${pic%.CR2}.jpg)";
done
