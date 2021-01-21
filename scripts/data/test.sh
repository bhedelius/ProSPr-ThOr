#!/bin/bash

print() {
    echo $1
}

export -f print

parallel print ::: {1..1000}
