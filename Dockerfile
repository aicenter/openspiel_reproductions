FROM waltonmyke/openspiel
RUN pip install -r requirements.txt
RUN pip install --upgrade ipython