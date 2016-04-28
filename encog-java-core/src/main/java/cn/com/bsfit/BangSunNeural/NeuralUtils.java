package cn.com.bsfit.BangSunNeural;

import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.persist.EncogDirectoryPersistence;

import java.io.*;

/**
 * Created by qiaoyang on 16-4-27.
 */
class NormalizationUtils {

    public static void persistModel(NormalizationHelper helper, String ModelPath) {
        ObjectOutputStream oos = null;
        try {
            oos = new ObjectOutputStream(new FileOutputStream(ModelPath));
            oos.writeObject(helper);
            oos.flush();
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // EncogDirectoryPersistence.saveObject(new File(ModelPath),helper);
    }

    public static NormalizationHelper reloadPersistModel(String ModelPath) {
        ObjectInputStream ois = null;
        try {
            ois = new ObjectInputStream(new FileInputStream(new File(ModelPath)));
            NormalizationHelper helper = (NormalizationHelper) ois.readObject();
            return helper;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void dataToFile(VersatileMLDataSet data, String outputFile) {

        BufferedWriter bw = null;
        int size = data.size();

        try {
            bw = new BufferedWriter(new FileWriter(outputFile));

            for(int i=0;i<size;i++){
                StringBuffer tempBuffer = new StringBuffer();
                String temp = data.get(i).toString();
                int index1 = temp.indexOf("[BasicMLDataPair:Input:[BasicMLData:");
                int index2 = temp.indexOf("]Ideal:[BasicMLData:");
                int index3 = temp.indexOf("],Significance:100.000000%]");
                tempBuffer.append(temp.substring(index1+36,index2));
                tempBuffer.append(",");
                tempBuffer.append(temp.substring(index2+20,index3));
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



