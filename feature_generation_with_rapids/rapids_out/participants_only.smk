configfile: "/rapids/data/config.yaml"
include: "/rapids/rules/common.smk"
include: "/rapids/rules/preprocessing.smk"
rule all:
    input:
        expand("data/external/participant_files/{pid}.yaml", pid=config.get("PIDS", []))
