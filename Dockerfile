FROM waltonmyke/openspiel
RUN apt-get update
RUN apt-get install -qqy x11-apps
RUN apt-get update && apt-get install -qqy x11-apps && apt-get install -qqy xterm
RUN useradd -ms /bin/bash xterm
USER xterm
WORKDIR /home/xterm
