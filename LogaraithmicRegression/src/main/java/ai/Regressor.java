package ai;

import java.util.LinkedList;
import java.util.List;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.DefaultRealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
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
		return  -1*(y*Math.log(y_)-(1-y)*Math.log(1-y_));
	}
	
	public double crossEntropyCost(RealMatrix w) {
		double cost=0;		
		RealMatrix m = hypothesis(w);
		m.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
	        @Override
	        public double visit(int row, int column, double value) {
	            return crossEntropyLoss(y.getEntry(row),value);
	        }
	    });
		cost=m.multiply(MatrixUtils.createRealMatrix(m.getColumnDimension(), m.getRowDimension()).scalarAdd(1)).getEntry(0,0);
		cost=(cost/y.getDimension());
		return cost;
	}
	
	public RealMatrix hypothesis(RealMatrix w) {
		RealMatrix temp=w.transpose().multiply(x);
		temp.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
	        @Override
	        public double visit(int row, int column, double value) {
	            return new Sigmoid().value(value);
	        }
	    });
		return temp;
	}
	
	public RealMatrix hypothesis() {
		return hypothesis(w);
	}
	
	public RealMatrix CrossEntropyCostMatrix(RealMatrix w) {
		RealMatrix xCalculate = x.copy();
		RealMatrix yMatrix= toMatrix(y);
		
		yMatrix=yMatrix.scalarMultiply(-1);
		yMatrix=yMatrix.add(hypothesis(w));
		yMatrix=yMatrix.transpose();
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
}
