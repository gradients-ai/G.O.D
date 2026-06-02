#!/bin/bash

if [ -z "$GRAFANA_USERNAME" ]; then
  GRAFANA_USERNAME=admin
  sed -i '/GRAFANA_USERNAME/d' .vali.env
  echo GRAFANA_USERNAME=$GRAFANA_USERNAME >> .vali.env
fi

GRAFANA_PASSWORD=$(grep GRAFANA_PASSWORD .vali.env | cut -d= -f2)

if [ -z "$GRAFANA_PASSWORD" ]; then
  GRAFANA_PASSWORD=$(openssl rand -hex 16)
  sed -i '/GRAFANA_PASSWORD/d' .vali.env
  echo GRAFANA_PASSWORD=$GRAFANA_PASSWORD >> .vali.env
fi
