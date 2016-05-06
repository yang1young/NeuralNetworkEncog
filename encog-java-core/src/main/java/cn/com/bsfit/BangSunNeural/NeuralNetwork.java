package cn.com.bsfit.BangSunNeural;


import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.XMLConfiguration;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.encog.Encog;
import org.encog.ml.MLClassification;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.TrainingSetUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by qiaoyang on 16-4-27.
 */

public class NeuralNetwork {

    private static XMLConfiguration brainConfig;
    protected static Logger logger = Logger.getLogger(NeuralNetwork.class);

    public NeuralNetwork(String configFile) {
        try {
            brainConfig = new XMLConfiguration(configFile);

        } catch (ConfigurationException e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) {
        PropertyConfigurator.configure("/home/qiaoyang/BangSun/encog-java-core/log4j.properties");
        NeuralNetwork network = new NeuralNetwork("/home/qiaoyang/BangSun/encog-java-core/src/main/resources/NeuralNetConf.xml");

        MLDataSet data = network.dataConfig(false,true);
        network.buildModel((VersatileMLDataSet) data);

        MLDataSet testData = network.dataConfig(false,false);

        network.reloadModel("NueralNetwork0.txt",testData);

    }


    public MLDataSet dataConfig(boolean hasHead,boolean isTrain) {

        String helperPath = brainConfig.getString("NormalizeHelperPath");
        String normalizeData = brainConfig.getString("NormalizeDataSet");
        int columnsLength = Integer.parseInt(brainConfig.getString("DataNominalIndex").split(":")[0]);

        if(isTrain){
            String originData = brainConfig.getString("TrainDataSet");
            String[] nomialIndexString = brainConfig.getString("DataNominalIndex").split(":")[1].split("/");
            Object[] options = brainConfig.getList("ModelParameters.Parameter").toArray();
            String EncogModelType = options[0].toString().split("/")[0];

            VersatileDataSource source = new CSVDataSource(new File(originData), hasHead, CSVFormat.DECIMAL_POINT);
            VersatileMLDataSet data = new VersatileMLDataSet(source);
            ColumnDefinition outputColumn;
            ArrayList<Integer> nomialIndex = new ArrayList<Integer>();
            for(int i = 0; i < nomialIndexString.length; i++){
                nomialIndex.add(new Integer(nomialIndexString[i]));
            }

            for (int j = 0; j < columnsLength - 1; j++) {
                if (!nomialIndex.contains(j)) {
                    data.defineSourceColumn("" + j, j, ColumnType.continuous);
                }else {
                    data.defineSourceColumn("" + j, j, ColumnType.nominal);
                }
            }

            if (nomialIndex.contains(columnsLength - 1)) {
                outputColumn = data.defineSourceColumn("label", columnsLength - 1, ColumnType.nominal);
            } else {
                outputColumn = data.defineSourceColumn("label", columnsLength - 1, ColumnType.continuous);

            }
            data.defineSingleOutputOthersInput(outputColumn);
            data.analyze();

            EncogModel model = new EncogModel(data);
            model.selectMethod(data, EncogModelType);
            data.normalize();

            NeuralUtils.dataToFile(data, normalizeData);
            NormalizationHelper helper = data.getNormHelper();
            logger.info(helper.toString());
            NeuralUtils.persistHelper(helper, helperPath);

            return data;
        }
        else{
            String testData = brainConfig.getString("TestDataSet");
            MLDataSet data = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH,testData, false, columnsLength-1, 1);
           // VersatileDataSource source = new CSVDataSource(new File(testData), hasHead, CSVFormat.DECIMAL_POINT);
           // VersatileMLDataSet data = new VersatileMLDataSet(source);

            return data;
        }
    }


    public void buildModel(VersatileMLDataSet data) {

        String helperPath = brainConfig.getString("NormalizeHelperPath");
        Object[] Options = brainConfig.getList("ModelParameters.Parameter").toArray();
        Object[] modelTypes = brainConfig.getList("ModelTypes.ModelType").toArray();
        Object[] modelNames = brainConfig.getList("ModelNames.ModelName").toArray();

        String crossValidateConf = brainConfig.getString("crossValidateConf");
        String modelPath = brainConfig.getString("ModelPath");

        NormalizationHelper helper = NeuralUtils.reloadHelper(helperPath);
       /* int inputNum = helper.getInputColumns().size();
        int outputNum = helper.getOutputColumns().size();
        MLDataSet data = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH,normalizeData, false, inputNum, outputNum);
*/
        for (int i = 0; i < modelTypes.length; i++) {

            int modelIndex = Integer.parseInt(modelTypes[i].toString());
            String[] options = Options[modelIndex].toString().split("/");
            String trainArgs = options[3].toString().replace('#', ',');
            options[3] = trainArgs;

            NeuralModels model = new NeuralModels(data, options);
            MLRegression bestModel = model.crossValidate(crossValidateConf);
            NeuralUtils.evalModelTrain(bestModel,data,helper);

            NeuralUtils.persistModel(bestModel,modelPath+modelNames[modelIndex]+i+".txt");

            Encog.getInstance().shutdown();

        }
    }


    public void reloadModel(String modelName,MLDataSet data){

        MLRegression model;
        String modelPath = brainConfig.getString("ModelPath");
        String helperPath = brainConfig.getString("NormalizeHelperPath");

        model = NeuralUtils.reloadModels(modelPath+modelName);
        NormalizationHelper helper  = NeuralUtils.reloadHelper(helperPath);

        NeuralUtils.evalModel(model,data,helper);
        String [] audit = "-0.99851,-0.998981,0,2009,-1,4,-1,-1,-1,1,-0.992481,0,-1,-1,-1,-1,-1,-1,-0.992481".split(",");
        System.out.println(NeuralUtils.audit(model,audit,helper));

    }

}



