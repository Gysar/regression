package main;

import java.awt.Color;
import java.io.IOException;
import java.io.ObjectInputStream.GetField;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import ai.Regressor;
import data.DataCreator;
import plot.Plot;
import plot.Plot.AxisFormat;

public class Main {

	public static void main(String[] args) {
		final int DATA_SET_COUNT = 100;
		final int ITERATION_COUNT = 2000;
		final int STEPS_PER_SAFE= 100;
		
		RealMatrix x= DataCreator.getData(DATA_SET_COUNT);
		double[][] line = DataCreator.getRandomLine(x);
		RealVector y = DataCreator.evaluate(line, x);
		for(double i=1;i>=0.001;i*=0.1) {
			learn(ITERATION_COUNT, STEPS_PER_SAFE, x, y, i);
		}
	}



	private static void learn(final int ITERATION_COUNT, final int STEPS_PER_SAFE, RealMatrix x, RealVector y, double alpha) {
		Regressor regressor = new Regressor(x, y, alpha);
		List<RealMatrix> weightHistory=regressor.logisticRegression(ITERATION_COUNT, STEPS_PER_SAFE);
		List<Double> iterations = new LinkedList<Double>();
		List<Double> middleCosts = new LinkedList<Double>();
		
		for(int i = 0; i<weightHistory.size();i++) {
			middleCosts.add(regressor.middleCrossEntropyCost(weightHistory.get(i)));
			iterations.add((double) ((int)(i*STEPS_PER_SAFE)));
		}
		String name=String.format("plot %.6f",alpha);
		plot(middleCosts,iterations,name);
	}
	
	
	
	private static void plot(List<Double> costs, List<Double> iterations, String name) {
		double min=Integer.MAX_VALUE;
		double max= Integer.MIN_VALUE;
		for(double cost:costs) {
			if(cost<min)min=cost;
			if(cost>max)max=cost;
		}
		Plot plot = Plot.plot(Plot.plotOpts().
		        title("costs").
		        legend(Plot.LegendFormat.BOTTOM)).
		    xAxis("iterations", Plot.axisOpts()
		    		.format(AxisFormat.NUMBER_INT).
		        range(0, iterations.get(iterations.size()-1))).
		    
		    yAxis("costs", Plot.axisOpts().
		    	format(AxisFormat.NUMBER).
		        range(min,max)).
		    
		    series("Data", Plot.data().xy(iterations, costs),
		        Plot.seriesOpts().
		            marker(Plot.Marker.NONE).
		            markerColor(Color.GREEN).
		            color(Color.BLACK));

		try {
			plot.save(name, "png");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
