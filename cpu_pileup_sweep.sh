#!/bin/bash
source /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc
export LD_LIBRARY_PATH=/data/alice/sbetisor/traccc-jp/build/lib64:$LD_LIBRARY_PATH
BIN=/data/alice/sbetisor/traccc-jp/build/bin/traccc_benchmark_resolver
OUTFILE=/data/alice/sbetisor/results/cpu_pileup_sweep_$(date +%Y%m%d_%H%M%S).txt
mkdir -p /data/alice/sbetisor/results
echo "CPU greedy vs JP pileup sweep" > $OUTFILE
echo "Date: $(date)" >> $OUTFILE
echo "Node: $(hostname)" >> $OUTFILE
echo "" >> $OUTFILE

for mu in 0 20 50 100 140 200 300 400 500 600; do
  DUMP=/data/alice/sbetisor/data/fatras_csv_dumps/fatras_ttbar_mu${mu}/event_000.json
  if [ ! -f "$DUMP" ]; then echo "SKIP mu=$mu (no dump)" | tee -a $OUTFILE; continue; fi

  echo "=== mu=$mu 1T ===" | tee -a $OUTFILE
  $BIN --input-dump=$DUMP --repeats=20 --warmup=5 --conflict-graph=jp 2>&1 | tee -a $OUTFILE

  for T in 2 4 16 32; do
    echo "=== mu=$mu greedy ${T}T ===" | tee -a $OUTFILE
    $BIN --input-dump=$DUMP --repeats=20 --warmup=5 --threads=$T 2>&1 | tee -a $OUTFILE
    echo "=== mu=$mu jp ${T}T ===" | tee -a $OUTFILE
    $BIN --input-dump=$DUMP --repeats=20 --warmup=5 --conflict-graph=jp --threads=$T 2>&1 | tee -a $OUTFILE
  done
done

echo "DONE" | tee -a $OUTFILE
echo "Results saved to $OUTFILE"
