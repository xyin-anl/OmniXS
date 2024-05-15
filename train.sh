#!/bin/bash

max_epochs=1

run() {
    python main.py compound_name="$1" simulation_type="$2" trainer.max_epochs="$max_epochs" model_name="$3"
}

run "$1" "$2" "$3"