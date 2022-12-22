FROM nvidia/cuda:11.4.0-base-ubuntu20.04
RUN echo 'Printing nvidia-smi below:'
CMD nvidia-smi

ENV PATH=“${PATH}:/root/.local/bin” 

RUN echo 'Exposing port 8090 for using the streamlit app'
EXPOSE 8090

COPY requirements.txt requirements.txt
RUN echo 'Upgrading pip and installing custom requirements'
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip

RUN pip install -r requirements.txt
WORKDIR /src
# Copy the relevant files over to the docker image
RUN echo 'Copying local code over to docker container'
COPY . /src
ENTRYPOINT [ "streamlit", "run" ]
CMD [ "streamlit_app.py", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false", "--server.port=8090", "--server.address=0.0.0.0"]