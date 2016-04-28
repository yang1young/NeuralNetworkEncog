package normalizers;

import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.columns.ColumnDefinition;

import java.io.Serializable;

/**
 * Created by qiaoyang on 16-4-28.
 */
public interface Normalizer extends Serializable{
    int outputSize(ColumnDefinition var1);
    int normalizeColumn(ColumnDefinition var1, String var2, double[] var3, int var4);
    int normalizeColumn(ColumnDefinition var1, double var2, double[] var4, int var5);
    String denormalizeColumn(ColumnDefinition var1, MLData var2, int var3);
}
