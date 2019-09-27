#!/bin/bash

for f in ls *.py
do 
	echo "processing $f"
	pylint $f
done
