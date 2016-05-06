package cn.com.bsfit.BangSunNeural;

import org.apache.log4j.Logger;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.persist.EncogDirectoryPersistence;

import java.io.*;

/**
 * Created by qiaoyang on 16-4-27.
 */
public class NeuralUtils {

    protected static Logger logger = Logger.getLogger(NeuralUtils.class);

    public static void persistHelper(NormalizationHelper helper, String HelperPath) {
        ObjectOutputStream oos = null;
        try {
            oos = new ObjectOutputStream(new FileOutputStream(HelperPath));
            oos.writeObject(helper);
            oos.flush();
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // EncogDirectoryPersistence.saveObject(new File(ModelPath),helper);
    }

    public static NormalizationHelper reloadHelper(String HelperPath) {
        ObjectInputStream ois = null;
        try {
            ois = new ObjectInputStream(new FileInputStream(new File(HelperPath)));
            NormalizationHelper helper = (NormalizationHelper) ois.readObject();
            return helper;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

  /*  public static void persistModel(MLRegression model, String ModelPath) {

        EncogDirectoryPersistence.saveObject(new File(ModelPath),model);
    }


    public static MLRegression reloadModels(String ModelPath){

        MLRegression model  =(MLRegression) EncogDirectoryPersistence.loadObject(new File(ModelPath));

        return model;
    }*/

    public static void persistModel(MLRegression helper, String HelperPath) {
        ObjectOutputStream oos = null;
        try {
            oos = new ObjectOutputStream(new FileOutputStream(HelperPath));
            oos.writeObject(helper);
            oos.flush();
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static MLRegression reloadModels(String HelperPath) {
        ObjectInputStream ois = null;
        try {
            ois = new ObjectInputStream(new FileInputStream(new File(HelperPath)));
            MLRegression helper = (MLRegression) ois.readObject();
            return helper;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }


    public static int audit(MLRegression bestModel,String[] auditData, NormalizationHelper helper) {

        int result;
        MLData input = helper.allocateInputVector();
        helper.normalizeInputVector(auditData, input.getData(), true);
        MLData output = bestModel.compute(input);
        result = Integer.parseInt(helper.denormalizeOutputVectorToString(output)[0]);

        return result;
    }

    public static void evalModel(MLRegression bestModel,MLDataSet valData, NormalizationHelper helper) {


        int[] report = new int[]{0, 0, 0, 0, 0, 0};
        System.out.println(helper.toString());
        for (MLDataPair pair : valData) {

            MLData input = pair.getInput();
            MLData actual = pair.getIdeal();
            int origin = (int) actual.getData()[0];

            int length = input.getData().length;
            String auditData[] = new String[length];

            for (int i = 0; i < length; i++) {
                double tempData = input.getData()[i];
                int m = (int)tempData;
                if(m==tempData){
                    auditData[i] = "" +m ;
                }else{
                    auditData[i] = "" +tempData ;
                }
            }

            int result = audit(bestModel, auditData, helper);

            if (origin == 0) {
                report[0]++;
                if (result == 0) {
                    report[2]++;
                } else {
                    report[3]++;
                }
            } else {
                report[1]++;
                if (result == 1) {
                    report[4]++;
                } else {
                    report[5]++;
                }
            }
        }

        logger.info("---------0's number is -----" + report[0]);
        logger.info("---------1's number is -----" + report[1]);

        logger.info("---------0->0-----" + report[2] + "-------Percentage = " + ((double) report[2]) / report[0]);
        logger.info("---------0->1-----" + report[3] + "-------Percentage = " + ((double) report[3]) / report[0]);
        logger.info("---------1->1-----" + report[4] + "-------Percentage = " + ((double) report[4]) / report[1]);
        logger.info("---------1->0-----" + report[5] + "-------Percentage = " + ((double) report[5]) / report[1]);

        logger.info("----------0's---------Precision--------" + ((double) report[2]) / (report[2] + report[5]));
        logger.info("----------0's---------Recall-------" + ((double) report[2]) / report[0]);
        logger.info("----------1's---------Precision-------" + ((double) report[4]) / (report[3] + report[4]));
        logger.info("----------1's---------Recall---------" + ((double) report[4]) / report[1]);
    }


    public static void evalModelTrain(MLRegression bestModel, MLDataSet valData, NormalizationHelper helper) {


        int[] report = new int[]{0, 0, 0, 0, 0, 0};

        for (MLDataPair pair : valData) {

            MLData input = pair.getInput();
            MLData actual = pair.getIdeal();

            MLData output = bestModel.compute(input);
            int result = Integer.parseInt(helper.denormalizeOutputVectorToString(output)[0]);
            int origin = Integer.parseInt(helper.denormalizeOutputVectorToString(actual)[0]);

            if (origin == 0) {
                report[0]++;
                if (result == 0) {
                    report[2]++;
                } else {
                    report[3]++;
                }
            } else {
                report[1]++;
                if (result == 1) {
                    report[4]++;
                } else {
                    report[5]++;
                }
            }
        }
        logger.info("---------0's number is -----" + report[0]);
        logger.info("---------1's number is -----" + report[1]);

        logger.info("---------0->0-----" + report[2] + "-------Percentage = " + ((double) report[2]) / report[0]);
        logger.info("---------0->1-----" + report[3] + "-------Percentage = " + ((double) report[3]) / report[0]);
        logger.info("---------1->1-----" + report[4] + "-------Percentage = " + ((double) report[4]) / report[1]);
        logger.info("---------1->0-----" + report[5] + "-------Percentage = " + ((double) report[5]) / report[1]);

        logger.info("----------0's---------Precision--------" + ((double) report[2]) / (report[2] + report[5]));
        logger.info("----------0's---------Recall-------" + ((double) report[2]) / report[0]);
        logger.info("----------1's---------Precision-------" + ((double) report[4]) / (report[3] + report[4]));
        logger.info("----------1's---------Recall---------" + ((double) report[4]) / report[1]);
    }


    public static void dataToFile(VersatileMLDataSet data, String outputFile) {

        BufferedWriter bw = null;
        int size = data.size();

        try {
            bw = new BufferedWriter(new FileWriter(outputFile));

            for (int i = 0; i < size; i++) {
                StringBuffer tempBuffer = new StringBuffer();
                String temp = data.get(i).toString();
                int index1 = temp.indexOf("[BasicMLDataPair:Input:[BasicMLData:");
                int index2 = temp.indexOf("]Ideal:[BasicMLData:");
                int index3 = temp.indexOf("],Significance:100.000000%]");
                tempBuffer.append(temp.substring(index1 + 36, index2));
                tempBuffer.append(",");
                tempBuffer.append(temp.substring(index2 + 20, index3));
                bw.write(tempBuffer.toString());
                bw.newLine();

            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                bw.flush();
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}

       /*String s;
            int origin=0;
            int result=0;
            try {
            BufferedReader br =new BufferedReader(new FileReader("/home/qiaoyang/BangSun/encog-java-core/src/main/resources/testData/data.csv"));
            do {
                s = br.readLine();
                String[]dataAll =s.split(",");
                String[] auditData = new String[dataAll.length-1];
                origin = Integer.parseInt(dataAll[dataAll.length-1]);
                for(int j =0;j<dataAll.length-1;j++) {
                    auditData[j] = dataAll[j];
                }
                result = audit(bestModel,auditData,helper);

                if (origin == 0) {
                    report[0]++;
                    if (result == 0) {
                        report[2]++;
                    } else {
                        report[3]++;
                    }
                } else {
                    report[1]++;
                    if (result == 1) {
                        report[4]++;
                    } else {
                        report[5]++;
                    }
                }
            } while (s != null);
        } catch (Exception e) {
            e.printStackTrace();
        }
*/

