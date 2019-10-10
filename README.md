# Bluestoned

Find bluescreen chroma keys or other color rangers in video files and images using OpenCV

## Quickstart

```sh
$ bluestoned -s video.mp4
```

Example output:

![](https://i.imgur.com/ixGO5Z2.png)

## Install

Simple install:

```sh
$ pip install -U bluestoned
```

There are some prerequisites for OpenCV

Debian / Ubuntu

```sh
$ sudo apt-get install libsm6 libxrender1 libfontconfig1
```

CentOS / RHEL

```sh
$ sudo yum install libXext libSM libXrender
```