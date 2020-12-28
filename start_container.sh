#!/bin/bash
HASH=$(cat /dev/urandom | env LC_CTYPE=C tr -dc a-zA-Z0-9 | head -c 4)

if [ -z "$1" ]
then
	name=openspiel_"$HASH"
else	
	name=openspiel_"$1"_"$HASH"
fi

docker run --name $name -d -t -p 8888:8888 --rm -v $(pwd):/notebooks waltonmyke/openspiel
docker exec $name pip3 install jupyter
docker exec $name pip3 install --upgrade ipython
docker exec $name pip3 install pandas
docker exec $name jupyter notebook --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=/notebooks
