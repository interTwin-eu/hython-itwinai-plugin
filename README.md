# EURAC Use Case

[![GitHub Super-Linter](https://github.com/interTwin-eu/itwinai-plugin-template/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/itwinai-plugin-template/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
 [![SQAaaS source code](https://github.com/EOSC-synergy/itwinai-plugin-template.assess.sqaaas/raw/main/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/itwinai-plugin-template.assess.sqaaas/main/.report/assessment_output.json)

**Integration Authors**: Jarl Sondre Sæther (CERN), Henry Mutegeki (CERN), Iacopo Ferrario
(EURAC), Matteo Bunino (CERN)

## Developer Installation

To install this package, use the following command: 

```bash
pip install -e .
```

### Installation of Horovod and DeepSpeed

If you are on JSC, you need to run a SLURM script to properly install Horovod and
DeepSpeed:

```bash
sbatch installation-scripts/horovod-deepspeed-JSC.slurm
```
The script will install Horovod and DeepSpeed with the correct installation flags. It
usually takes around 20 minutes to complete. 

## Launching the Training

You can launch the training using `itwinai`'s `exec-pipeline` command as follows:

```bash
itwinai exec-pipeline --config-path configuration_files --config-name training
```
