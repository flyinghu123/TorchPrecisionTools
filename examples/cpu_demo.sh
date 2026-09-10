export TPD_ENABLED=1
export TPD_OUTPUT_DIR=saves/result1
export TPD_SAMPLE_COUNT=3
export MAX_STEPS=2

python cpu_demo.py


# tpd compare saves/result0 saves/result1 -o saves/comparison.json
# tpd stack saves/result0 S000005_a36a855ab094
# tpd cmp summary saves/comparison.json