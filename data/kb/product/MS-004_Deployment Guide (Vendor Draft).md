# Deployment Guide

This document is provided by vendor for internal reference only. 部分内容待确认。

## 1. Prerequisites

- Docker 24+ / docker-compose v2

- 4 CPU cores minimum, 16GB RAM recommended

- PostgreSQL 14（我们用的是 14.5，客户环境建议 14 以上）

## 2. Steps

1. Unzip the package to /opt/beichen

2. 修改 .env 文件

3. docker-compose up -d

4. 等待 health check 通过 (约3分钟)

## 3. Known Issues

- 如果 5432 端口被占用，容器会起不来。改 .env 里的 PORT。

- 首次启动可能会因为镜像拉取慢超时，可以把镜像先 load 进去。

- TODO: 补充升级步骤 (待 vendor 确认)
