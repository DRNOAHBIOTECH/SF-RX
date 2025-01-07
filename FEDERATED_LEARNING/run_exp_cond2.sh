#!/bin/bash

#각 조건에 대한 설정
FL_END_ROUND=31
CONDITION=2

for LV in {1..4}; do
    python3 experiment.py --fl_end_round "$FL_END_ROUND" --lv "$LV" --condition "$CONDITION"&
done

wait