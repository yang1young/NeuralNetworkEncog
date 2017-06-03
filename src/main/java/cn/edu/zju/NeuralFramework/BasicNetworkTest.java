package cn.edu.zju.NeuralFramework;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.folded.FoldedDataSet;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.model.EncogModel;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.ContainsFlat;
import org.encog.neural.networks.training.cross.CrossValidationKFold;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.pattern.ElmanPattern;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.TrainingSetUtil;

import java.io.File;


/**
 * Created by qiaoyang on 16-4-27.
 */

public class BasicNetworkTest {

    public static void main(String[] args) {

        BasicNetworkTest hah = new BasicNetworkTest();
        // hah.dataConfig();
        // hah.reload();
        BasicNetwork network = hah.network();
        System.out.println("*********************************");
        hah.train(network);
        System.out.println("*********************************");

        hah.evaluate(new File("/home/qiaoyang/BangSun/encog-java-core/src/main/resources/model/neural.txt"));
        System.out.println("*********************************");

        hah.predict(network);
    }

    public void dataConfig()
    {
        VersatileDataSource source = new CSVDataSource(new File("/home/qiaoyang/BangSun/encog-java-core/src/main/resources/trainData/iris.csv"), false,
                CSVFormat.DECIMAL_POINT);
        VersatileMLDataSet data = new VersatileMLDataSet(source);
        data.defineSourceColumn("sepal-length", 0, ColumnType.continuous);
        data.defineSourceColumn("sepal-width", 1, ColumnType.continuous);
        data.defineSourceColumn("petal-length", 2, ColumnType.continuous);
        data.defineSourceColumn("petal-width", 3, ColumnType.continuous);

        // Define the column that we are trying to predict.
        ColumnDefinition outputColumn = data.defineSourceColumn("species", 4,
                ColumnType.nominal);
        data.analyze();
        data.defineSingleOutputOthersInput(outputColumn);

        EncogModel model = new EncogModel(data);

        model.selectMethod(data, "feedforward");
        data.normalize();
        System.out.println(data.get(0).toString());
        NeuralUtils.dataToFile(data, "/home/qiaoyang/BangSun/encog-java-core/src/main/resources/trainData/test.csv");

        NormalizationHelper helper = data.getNormHelper();
        System.out.println(helper.toString());
        NeuralUtils.persistHelper(helper, "/home/qiaoyang/BangSun/encog-java-core/src/main/resources/Helper/helper.txt");

    }

    public BasicNetwork network() {
        ElmanPattern pattern = new ElmanPattern();
        pattern.setInputNeurons(4);
        pattern.addHiddenLayer(10);
        pattern.setOutputNeurons(3);
        pattern.setActivationFunction(new ActivationSigmoid());
        BasicNetwork network = (BasicNetwork) pattern.generate();
              /*  final BasicNetwork network = new BasicNetwork();
                network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 43));
                network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 32));

                network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 2));
                network.getStructure().finalizeStructure();
                network.reset();*/
        return network;
    }

    public void train(ContainsFlat network) {
        MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(
                CSVFormat.ENGLISH, "/home/qiaoyang/BangSun/encog-java-core/src/main/resources/trainData/test.csv", false, 4, 3);

        final FoldedDataSet folded = new FoldedDataSet(trainingSet);
        final MLTrain train = new ResilientPropagation(network, folded);
        final CrossValidationKFold trainFolded = new CrossValidationKFold(train, 5);

        int epoch = 1;

        do {
            trainFolded.iteration();
            System.out
                    .println("Epoch #" + epoch + " Error:" + trainFolded.getError());
            epoch++;
        } while (trainFolded.getError() > 1);

        EncogDirectoryPersistence.saveObject(new File("/home/qiaoyang/BangSun/NeuralNetwork/src/main/resources/model/neural.txt"), network);


    }

    public void evaluate(File file) {


        MLDataSet data = TrainingSetUtil.loadCSVTOMemory(
                CSVFormat.ENGLISH, "/home/qiaoyang/BangSun/NeuralNetwork/src/main/resources/trainDataSet/test.csv", false, 4, 3);

        if (!file.exists()) {
            System.out.println("Can't read file: " + file.getAbsolutePath());
            return;
        }
        BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(file);
        NormalizationHelper helper = NeuralUtils.reloadHelper("/home/qiaoyang/BangSun/encog-java-core/src/main/resources/Helper/helper.txt");

        int count = 0;
        int correct = 0;

        for (MLDataPair pair : data) {

            MLData input = pair.getInput();
            MLData actual = pair.getIdeal();
            MLData output = network.compute(input);
            String irisChosen = helper.denormalizeOutputVectorToString(output)[0];

            String origin = helper.denormalizeOutputVectorToString(actual)[0];

            if (irisChosen.equals(origin))
                correct++;
            count++;

        }
        double percent = (double) correct / (double) count;
        System.out.println("Direction correct:" + correct + "/" + count);
        System.out.println("Directional Accuracy:"
                + percent * 100 + "%");
    }

    public void predict(BasicNetwork network) {
        NormalizationHelper helper = NeuralUtils.reloadHelper("/home/qiaoyang/BangSun/encog-java-core/src/main/resources/Helper/helper.txt");
        System.out.println(helper.toString());
        String[] line = {"5.1", "3.5", "1.4", "0.2", "Iris-setosa"};
        MLData input = helper.allocateInputVector();
        String correct = line[4];
        helper.normalizeInputVector(line, input.getData(), false);
        MLData output = network.compute(input);
        String irisChosen = helper.denormalizeOutputVectorToString(output)[0];

        System.out.println(irisChosen);

        Encog.getInstance().shutdown();

    }


}

