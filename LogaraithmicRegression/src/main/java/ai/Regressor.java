package ai;

import java.util.LinkedList;
import java.util.List;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Regressor {
	RealMatrix x;
	RealVector y;
	RealMatrix w; 
	double alpha;
	
	public Regressor(RealMatrix x, RealVector y, RealMatrix w, double alpha) {
		this(x,y,w);
		this.alpha = alpha;
	}
	
	public Regressor(RealMatrix x, RealVector y, double alpha) {
		this(x,y);
		this.alpha=alpha;
	}
	
	public Regressor(RealMatrix x, RealVector y, RealMatrix w) {
		this(x,y);
		this.w=w;
	}
	
	public Regressor(RealMatrix x, RealVector y) {
		this.x = x;
		this.y = y;
		this.w = MatrixUtils.createRealMatrix(3,1);
//		this.w = MatrixUtils.createRealMatrix(new double[][]{{1},{1},{1}});
		alpha=0.1;
	}
	
	public List<RealMatrix> logisticRegression(int iterations, int stepsPerSafe) {
		List<RealMatrix> weightsHistory = new LinkedList<RealMatrix>();
		for(int i = 0; i< iterations; i++) {
			updateWeights();
			if(i%stepsPerSafe==0) {
				weightsHistory.add(w);
			}
		}
		weightsHistory.add(w);
		return weightsHistory;
	}
	
	private void updateWeights() {
		RealMatrix xCalculate = CrossEntropyCostMatrix();
		xCalculate=xCalculate.scalarMultiply(alpha);
		xCalculate=xCalculate.scalarMultiply(-1);
		w=w.add(xCalculate);
	}
	
	public double crossEntropyLoss(double y, double y_) {
		double num= -1*(y*Math.log(y_)+(1-y)*Math.log(1-y_));
		return num;
	}
	
	public double middleCrossEntropyCost(RealMatrix w) {
		double cost=0;
		RealVector v=new ArrayRealVector(y.getDimension(),hypothesis(w));
		cost=toMatrix(y).transpose().multiply(toMatrix(v)).getEntry(0, 0);
		return (cost/y.getDimension());
	}
	
	public double hypothesis(RealMatrix w) {
		double z=w.transpose().multiply(x).getEntry(0, 0);
		return sigmoid(z);
	}
	
	public double hypothesis() {
		return hypothesis(w);
	}
	
	public RealMatrix CrossEntropyCostMatrix(RealMatrix w) {
		RealMatrix xCalculate = x.copy();
		RealMatrix yMatrix= toMatrix(y);
		yMatrix=yMatrix.scalarAdd(-1*hypothesis(w));
		yMatrix=yMatrix.scalarMultiply(-1).transpose();
		xCalculate=xCalculate.multiply(yMatrix);
		double scalar=(double)1/y.getDimension();
		return xCalculate.scalarMultiply(scalar);
	}
	
	public RealMatrix CrossEntropyCostMatrix() {
		return CrossEntropyCostMatrix(w);
	}
	
	private RealMatrix toMatrix(RealVector vector) {
		double[][] data = new double[1][];
		data[0]=vector.toArray();
		return MatrixUtils.createRealMatrix(data);
	}
	
	public double sigmoid(double value) {
//		Sigmoid sigmoid = new Sigmoid();
//		return sigmoid.value(value);
		return (double)1/(1+Math.pow(Math.E,-1*value));
	}
	

}
