FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

RUN pip install cython
RUN pip install SimpleITK

WORKDIR /C/docker/base

COPY . .
