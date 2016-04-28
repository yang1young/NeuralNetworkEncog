package normalizers.strategies;

import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.columns.ColumnDefinition;

import java.io.Serializable;

public interface NormalizationStrategy extends Serializable {
    int normalizedSize(ColumnDefinition var1, boolean var2);
    int normalizeColumn(ColumnDefinition var1, boolean var2, String var3, double[] var4, int var5);
    String denormalizeColumn(ColumnDefinition var1, boolean var2, MLData var3, int var4);
    int normalizeColumn(ColumnDefinition var1, boolean var2, double var3, double[] var5, int var6);
}