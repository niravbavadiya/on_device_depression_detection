configfile: "/rapids/data/config.yaml"
include: "/rapids/rules/common.smk"
include: "/rapids/rules/preprocessing.smk"
include: "/rapids/rules/features.smk"
_pids = config.get("PIDS") or []
rule all:
    input:
        expand("data/processed/features/{pid}/all_sensor_features.csv", pid=_pids)
