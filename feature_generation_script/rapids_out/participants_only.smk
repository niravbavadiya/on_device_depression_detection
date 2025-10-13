# Only what we need to create participant YAMLs
configfile: "/rapids/data/config.yaml"
include: "/rapids/rules/common.smk"
include: "/rapids/rules/preprocessing.smk"

# Build a YAML per PID when you run snakemake
