#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd ${SCRIPT_DIR}/../data

wget -O elasticsearch-8.15.5-darwin-aarch64.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.5-darwin-aarch64.tar.gz

# unzip
tar zxvf elasticsearch-8.15.5-darwin-aarch64.tar.gz
# rm elasticsearch-8.15.5-darwin-aarch64.tar.gz
cd elasticsearch-8.15.5

# fetch current path
current_path="$(pwd)"

# configuration
cat > config/elasticsearch.yml <<EOF
cluster.name: yuebaoqing
discovery.type: single-node
node.name: node-1
node.attr.rack: r1
path.data: ${current_path}/data
path.logs: ${current_path}/logs
network.host: 0.0.0.0
http.port: 9200
action.destructive_requires_name: false
xpack.security.enabled: false
EOF

# Start Elasticsearch
unset JAVA_HOME # to avoid using default java paths, which may bring some unexpected behavior
unset CLASSPATH
nohup bin/elasticsearch &

# wait for port 9200 to open
echo "Waiting for Elasticsearch to start..."
RETRIES=60
until curl -s http://localhost:9200 >/dev/null; do
  RETRIES=$((RETRIES - 1))
  if [ $RETRIES -le 0 ]; then
    echo "Elasticsearch failed to start within 60 seconds."
    exit 1
  fi
  sleep 1
done
echo "Elasticsearch is up!"
