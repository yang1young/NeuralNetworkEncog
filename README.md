# Framework of Neural Network based on Encog
* fix Serialization and neural network init,etc some bugs
* implement a framework to use Encog easier
* cofiguration based on XML and log based on log4j
* implement some tools to do data cleaning and saving


## Encog 3.3

https://github.com/encog/encog-java-core.git
The following links will be helpful getting started with Encog.
Getting Started:
http://www.heatonresearch.com/wiki/Getting_Started
Important Links:
http://www.heatonresearch.com/encog

## file introduce

1. cn.edu.zju.NeuralFramework/   my implement
2. org.encog/                    fix some bugs from Encog 3.3
3. BasicNetworkTest/IrisTest     tests using Encog
4. NeuralModels                  models here
5. NeuralNetwork                 main file, configure and train using framework
6. NeuralUtils                   tools to save and reload models, evaluation,etc

## using guide
1. read doc of Encog (NeuralNetworkFrameworkOfEncog/src/main/resources/doc/)
2. modify  NeuralNetConf.xml and set your parameters
3. modify NeuralNetwork main function then running

#### if you think this project may helps you, may you give me a star? :)