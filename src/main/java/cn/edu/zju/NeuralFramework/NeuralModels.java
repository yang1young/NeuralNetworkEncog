package cn.edu.zju.NeuralFramework;

import org.apache.log4j.Logger;
import org.encog.ConsoleStatusReportable;
import org.encog.ml.MLRegression;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.model.EncogModel;
import org.encog.util.simple.EncogUtility;


/**
 * Created by qiaoyang on 16-4-28.
 */
public class NeuralModels {

    protected static Logger logger = Logger.getLogger(NeuralModels.class);
    private EncogModel model;
    String[] options;

    public NeuralModels(VersatileMLDataSet data, String[] options) {

        if (options == null) {
            logger.info("Please set model options first!");
        } else {
            this.options = options;
            if (this.options.length < 4) {
                logger.info("Please completion model arguments!");
            }
        }
        model = new EncogModel(data);
        model.selectMethod(data, this.options[0], this.options[1], this.options[2], this.options[3]);
        model.setReport(new ConsoleStatusReportable());
    }

    public MLRegression crossValidate(String crossValidateConf) {

        MLRegression bestModel;
        String[] options = crossValidateConf.split("/");
        int fold = Integer.parseInt(options[0]);
        int seed = Integer.parseInt(options[1]);
        double percentage = Double.parseDouble(options[2]);
        int maxTolerance = Integer.parseInt(options[3]);
        double minStepImprove = Double.parseDouble(options[4]);
        model.setMaxTolerance(maxTolerance);
        model.setMinStepImprove(minStepImprove);
        model.holdBackValidation(percentage, true, seed);
        // Use a 5-fold cross-validated train.  Return the best method found.
        bestModel = (MLRegression) model.crossvalidate(fold, true);
        logger.info("Training error: " + EncogUtility.calculateRegressionError(bestModel, model.getTrainingDataset()));
        logger.info("Validation error: " + EncogUtility.calculateRegressionError(bestModel, model.getValidationDataset()));

        return bestModel;
    }
}
