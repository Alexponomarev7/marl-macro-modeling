configfile: "conf/config.yaml"

rule generate_data:
    shell:
        "python lib/dataset.py batch"