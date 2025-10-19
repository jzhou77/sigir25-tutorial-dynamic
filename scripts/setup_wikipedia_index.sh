mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd

scripts/start_elasticsearch.sh

# index wiki corpus
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
