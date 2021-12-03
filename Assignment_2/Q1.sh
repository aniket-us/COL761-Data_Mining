path=$(dirname $0)
graphs=$(python $path/Q1/format.py $1 $path/Q1/output/data-gspan gspan 2>&1)
graphs=$(python $path/Q1/format.py $1 $path/Q1/output/data-fsg fsg 2>&1)
graphs=$(python $path/Q1/format.py $1 $path/Q1/output/data-gaston gaston 2>&1)
python $path/Q1/plot.py $path/Q1/output/data-gspan $path/Q1/output/data-fsg $path/Q1/output/data-gaston $graphs $2
