FROM python:3.6

RUN useradd --create-home --home-dir /home/anonymous --shell /bin/bash anonymous

# Add the user `anonymous`
COPY . /home/anonymous/

# Install dependencies
RUN pip install -r /home/anonymous/requirements.txt

# Compile Cython code
RUN cd /home/anonymous/lib/model/_heavy/ && python setup.py build_ext --inplace

CMD jupyter notebook /home/anonymous/notebooks --NotebookApp.token=dummydummy --ip 0.0.0.0 --no-browser --allow-root --port 8888

EXPOSE 8888
