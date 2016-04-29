package cn.com.bsfit.BangSunNeural;


import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.commons.configuration.XMLConfiguration;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;

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
        network.dataConfig(false);
    }


    public void dataConfig(boolean hasHead) {

        String originData = brainConfig.getString("TrainDataSet");
        String normalizeData = brainConfig.getString("NormalizeDataSet");
        String helperPath = brainConfig.getString("NormalizeHelperPath");
        String[] nomialIndexString = brainConfig.getString("DataNominalIndex").split(":")[1].split(",");
        int columnsLength = Integer.parseInt(brainConfig.getString("DataNominalIndex").split(":")[0]);
        Object[] options =brainConfig.getList("ModelParameters.Parameter").toArray();
        String EncogModelType = options[0].toString();

        VersatileDataSource source = new CSVDataSource(new File(originData), hasHead, CSVFormat.DECIMAL_POINT);
        VersatileMLDataSet data = new VersatileMLDataSet(source);
        ColumnDefinition outputColumn;
        ArrayList<Integer> nomialIndex = new ArrayList<Integer>();

        for (int i = 0; i < nomialIndexString.length; i++) {
            nomialIndex.add(new Integer(nomialIndexString[i]));
            if (Integer.parseInt(nomialIndexString[i]) != columnsLength - 1) {
                data.defineSourceColumn(nomialIndexString[i], Integer.parseInt(nomialIndexString[i]), ColumnType.nominal);
            }
        }
        for (int j = 0; j < columnsLength-1; j++) {
            if (!nomialIndex.contains(j)) {
                data.defineSourceColumn("" + j, j, ColumnType.continuous);
            }
        }

        if (nomialIndex.contains(columnsLength - 1)) {
            outputColumn = data.defineSourceColumn("species", columnsLength - 1, ColumnType.nominal);
        } else {
            outputColumn = data.defineSourceColumn("species", columnsLength - 1, ColumnType.continuous);

        }
        data.defineSingleOutputOthersInput(outputColumn);
        data.analyze();

        EncogModel model = new EncogModel(data);
        model.selectMethod(data, EncogModelType);
        data.normalize();

        NeauralUtils.dataToFile(data, normalizeData);
        NormalizationHelper helper = data.getNormHelper();
        logger.info(helper.toString());
        NeauralUtils.persistHelper(helper, helperPath);

    }


}



