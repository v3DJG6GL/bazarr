# syntax=docker/dockerfile:1

FROM ghcr.io/linuxserver/unrar:latest AS unrar

# build the frontend from this checkout
FROM node:24-alpine AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM ghcr.io/linuxserver/baseimage-alpine:3.23

# set version label
ARG BUILD_DATE
ARG VERSION
ARG BAZARR_VERSION
LABEL build_version="v3DJG6GL version:- ${VERSION} Build-date:- ${BUILD_DATE}"
LABEL maintainer="v3DJG6GL"
# hard set UTC in case the user does not define it
ENV TZ="Etc/UTC"

# install requirements first so code changes don't invalidate this layer
COPY requirements.txt postgres-requirements.txt /app/bazarr/bin/

RUN \
  echo "**** install build packages ****" && \
  apk add --no-cache --virtual=build-dependencies \
    build-base \
    cargo \
    libffi-dev \
    libpq-dev \
    libxml2-dev \
    libxslt-dev \
    python3-dev && \
  echo "**** install packages ****" && \
  apk add --no-cache \
    ffmpeg \
    libxml2 \
    libxslt \
    mediainfo \
    python3 && \
  echo "**** Install requirements ****" && \
  python3 -m venv /lsiopy && \
  pip install -U --no-cache-dir \
    pip \
    wheel && \
  pip install -U --no-cache-dir --find-links https://wheel-index.linuxserver.io/alpine-3.23/ \
    -r /app/bazarr/bin/requirements.txt \
    -r /app/bazarr/bin/postgres-requirements.txt && \
  echo "**** clean up ****" && \
  apk del --purge \
    build-dependencies && \
  rm -rf \
    $HOME/.cache \
    $HOME/.cargo \
    /tmp/*

# install bazarr from this checkout (same file set as .github/files_to_copy)
COPY bazarr.py /app/bazarr/bin/
COPY bazarr/ /app/bazarr/bin/bazarr/
COPY custom_libs/ /app/bazarr/bin/custom_libs/
COPY libs/ /app/bazarr/bin/libs/
COPY migrations/ /app/bazarr/bin/migrations/
COPY --from=frontend /src/frontend/build/ /app/bazarr/bin/frontend/build/

RUN \
  echo "${BAZARR_VERSION:-unknown}" > /app/bazarr/bin/VERSION && \
  printf "UpdateMethod=docker\nBranch=master\nPackageVersion=${VERSION}\nPackageAuthor=v3DJG6GL\n" > /app/bazarr/package_info && \
  printf "v3DJG6GL version: ${VERSION}\nBuild-date: ${BUILD_DATE}\n" > /build_version

# add local files
COPY root/ /

# add unrar
COPY --from=unrar /usr/bin/unrar-alpine /usr/bin/unrar

# ports and volumes
EXPOSE 6767

VOLUME /config
