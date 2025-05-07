#!/bin/bash

FL_END_ROUND=31

for LV in {1..4}; do
    python3 experiment.py --fl_end_round "$FL_END_ROUND" --lv "$LV"&
done

wait