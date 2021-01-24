package data;
import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class DataCreator {
	/**
	 * each data point starts with 1 to make calculations easier. The second value is x, the third is y.
	 * @return
	 */
	public static RealMatrix getData(int number) {
		double[][] matrixData = new double[number][3];
		Random random = new Random();
		for(int i = 0; i<number; i++) {
			double x= random.nextDouble()*2-1;
			double y = random.nextDouble()*2-1;
			double[] row={1,x,y};
			matrixData[i]= row;
		}
		RealMatrix matrix = MatrixUtils.createRealMatrix(matrixData).transpose();
		return matrix;
	}

	
	public static RealVector evaluate(double[][] line, RealMatrix matrix) {
		double data[] = new double[matrix.getColumnDimension()];
		for(int i = 0; i<matrix.getColumnDimension(); i++) {
			data[i]=isRight(line[0],line[1],matrix.getColumn(i))?1:-1;
		}
		return new ArrayRealVector(data);
	}
	
	public static double[][] getRandomLine(RealMatrix matrix) {
		Random random = new Random();
		int p1= random.nextInt(matrix.getColumnDimension());
		int p2 = random.nextInt(matrix.getColumnDimension());
		double[] row1=matrix.getColumn(p1);
		double[] row2=matrix.getColumn(p2);
		double[][] line= {{row1[1],row1[2]},{row2[1],row2[2]}};
		return line;
	}
	
	private static boolean isRight(double[] linePoint1, double[] linePoint2 , double[] toCheck){
	     return ((linePoint2[0] - linePoint1[0])*(toCheck[2] - linePoint1[1]) - (linePoint2[1] - linePoint1[1])*(toCheck[1] - linePoint1[0])) <= 0;
	}
}
