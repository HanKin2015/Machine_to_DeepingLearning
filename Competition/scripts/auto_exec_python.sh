#!/bin/bash

while [ ! -z `pidof python` ]; do
    echo `date`
    sleep 10
done

python stacking.py
