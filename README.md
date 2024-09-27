# Privacy Evaluator
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Gitter](https://badges.gitter.im/fairlearn/community.svg)](https://gitter.im/fairlearn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

- space for the description of our project (core idea)

- For example:
The privML Privacy Evaluator is a tool that assesses ML model's levels of privacy by running different attacks on it. .....

## Dependencies

- [Python 3.7](https://www.python.org/)
- [Adversarial Robustness Toolbox (ART) 1.6.1](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [PyTorch 1.8.1](https://pytorch.org/)
- for detailed list see [requirements.txt](requirements.txt) 

## Installation
- if we need this section
- could be a command line code

## How to use the Privacy Evaluator?
- description how our project works


### Example
- some examples would be good


## Attacks

### Property Inference Attack
![plot](docs/Property_Interference_Attacks.png)

The Property Inference Attack aims to detect patterns in the parameters of the target model including properties 
which the model producer has not intended to reveal. Therefore, the adversary requires a 
meta-classifier which predicts whether one of these properties occurs in the target model or does not. 

The first step of the attack is to generate a set containing k datasets for k shadow classifiers which then, are trained 
to build the training set for the meta classifier. These datasets could be created by sampling from a larger dataset or
by integrating more data. Furthermore, the half of the datasets *D<sub>i<sub>* *(for i=1,...,k)* comprises a Property
distribution so that the Property **P** occurs and the other half another Property Distribution so that the 
Property does not occur. In addition, each shadow classifier comprises the same architecture as the target model 
where each is only fitted by its corresponding dataset. They do not need to perform as good as the target model,
but demonstrate passably acceptable performance.
Because the parameters in neural networks are usually randomly initialized and even after training, the order of 
parameters is arbitrary. Thus, the meta classifier must be trained on this so that it is able to recognize pattern between 
the parameters of the shadow classifiers. As a result, all the parameters of each shadow classifier represent the 
feature representation. The feature representation form together the training set for the meta classifer 
where each is labeled as Property **P** occurs in the target model or does not. The training algorithm for the 
meta-classifier can be arbitrarily chosen.

### Membership Inference Attack


## Getting Involved
If you want to contribute in any way, please visit our [Contribution Guidelines](./CONTRIBUTING.md) to get started.
Please have also a look at our [Code of Conduct](./CODE_OF_CONDUCT.md). 

### Contact
Gitter: https://gitter.im/privML/community
You can join our [Gitter](https://gitter.im/privML/community) communication channel.

### Who is behind the project?
- write the list of our supporting companies here
See a list of our [Supporters]().
- Zalando 
- ...

## License
This library is available under the [MIT License](https://github.com/git/git-scm.com/blob/master/MIT-LICENSE.txt).
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
-which is the correct link?

## References
- are there related projects?
- relevant books, papers etc.
