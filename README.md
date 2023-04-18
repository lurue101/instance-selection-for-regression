# kondo-ML

kondo-ML is a package containing various instance selection algorithms
usable with regression models. The implementations are compatible with sklearn and follow
its outlier detection interface.

## Install

The package can be installed via pip <br>
`pip install kondo_ml`

## Overview of algorithms

| Algorithm          | Goal                        |
|--------------------|-----------------------------|
| RegCNN             | Size reduction              |
| RegENN             | Noise filter                |
| RegENNTime         | Noise filter/drift handling |
| DROP-RX            | Noise filter/size reduction |
| Shapley            | Utility assignment          |
| FISH               | Drift Handling              |
| SELCON             | Size reduction              |
| Mutual Information | Noise filter                |
|                    |                             |

## Algorithm sources
RegCNN & RegENN: https://link.springer.com/chapter/10.1007/978-3-642-33266-1_33  <br>
DROP-RX: https://www.sciencedirect.com/science/article/abs/pii/S0925231216301953 <br>
Shapley: https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf  <br>
FISH: http://eprints.bournemouth.ac.uk/18567/1/FISH_journal_preprint.pdf <br>
SELCON: https://arxiv.org/abs/2106.12491 <br>
Mutual Information: https://research.cs.aalto.fi//aml/Publications/Publication167.pdf <br>

The SELCON implementation is taken from the author's github with minor changes: https://github.com/abir-de/SELCOn

## Example

Examples can be found in the notebooks of the examples folder in the github repository


## Contribution

Please feel free to contribute documentation, tests or new algorithms to this package.
And let me know if you find any mistakes in the implementations