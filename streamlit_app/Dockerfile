FROM python:3.8
# EXPOSE THE RELEVANT PORT FOR THE APP
RUN echo 'Exposing port 8083 for using the streamlit app'
EXPOSE 8083
COPY requirements.txt requirements.txt
# Upgrade pip and install requirements.txt
RUN echo 'Upgrading pip and installing custom requirements'
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy the relevant files over to the docker image
RUN echo 'Copying local code over to docker container'
COPY streamlit_app.py streamlit_app.py
COPY style.css style.css
COPY favicon.ico favicon.ico
COPY hf.png hf.png
# Set the working directory to the current directory
# Run the app
WORKDIR .
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8083", "--server.address=0.0.0.0"]
