package main;

import java.awt.Color;
import java.io.IOException;
import java.io.ObjectInputStream.GetField;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import ai.Regressor;
import data.DataCreator;
import plot.Plot;
import plot.Plot.AxisFormat;
import plot.Plot.Line;

public class Main {

	public static void main(String[] args) {
		final int DATA_SET_COUNT = 100;
		final int ITERATION_COUNT = 2000;
		final int STEPS_PER_SAFE= 100;
		
		RealMatrix x= DataCreator.getData(DATA_SET_COUNT);
		double[][] line = DataCreator.getRandomLine(x);
		RealVector y = DataCreator.evaluate(line, x);
		for(double i=1;i>=0.0001;i*=0.1) {
			learn(ITERATION_COUNT, STEPS_PER_SAFE, x, y, i);
		}
	}



	private static void learn(final int ITERATION_COUNT, final int STEPS_PER_SAFE, RealMatrix x, RealVector y, double alpha) {
		Regressor regressor = new Regressor(x, y, alpha);
		List<RealMatrix> weightHistory=regressor.logisticRegression(ITERATION_COUNT, STEPS_PER_SAFE);
		List<Double> iterations = new LinkedList<Double>();
		List<Double> middleCosts = new LinkedList<Double>();
		
		for(int i = 0; i<weightHistory.size();i++) {
			middleCosts.add(regressor.crossEntropyCost(weightHistory.get(i)));
			iterations.add((double) (i*STEPS_PER_SAFE));
		}

		plot(middleCosts,iterations,String.format("plot %.6f",alpha));
		plotPointsWithFunction(x,y,weightHistory,String.format("data %.6f",alpha));
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
		    	
		        range(min,max)).
		    
		    series("Data", Plot.data().xy(iterations, costs),
		        Plot.seriesOpts().
		            marker(Plot.Marker.NONE).
		            markerColor(Color.GREEN).
		            color(Color.BLACK));

		try {
			plot.save("res/"+name, "png");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private static double function(RealMatrix weights, double[] x) {		
		return weights.getEntry(0, 0)*x[0]+weights.getEntry(1, 0)*x[1]+weights.getEntry(2, 0)*x[2];
	}
	
	private static void plotPointsWithFunction(RealMatrix x, RealVector y, List<RealMatrix>weights,String name) {
		System.out.println(name+"   "+weights.get(weights.size()-1));
		double min= -2;
		double max= 2;
		double[] allx=x.getRow(1);
		double[] ally=x.getRow(2);
		List<Double> positivex=new ArrayList<Double>();
		List<Double> positivey=new ArrayList<Double>();
		List<Double> negativex=new ArrayList<Double>();
		List<Double> negativey=new ArrayList<Double>();
		for(int i=0;i<allx.length;i++) {
			if(y.getEntry(i)==1) {
				positivex.add(allx[i]);
				positivey.add(ally[i]);
			}else {
				negativex.add(allx[i]);
				negativey.add(ally[i]);
			}
		}
		
		Plot plot = Plot.plot(Plot.plotOpts().
		        title("points").
		        legend(Plot.LegendFormat.BOTTOM)).
		    xAxis("x1", Plot.axisOpts()
		    		.
		        range(min, max)).
		    
		    yAxis("x2", Plot.axisOpts().
		        range(min,max))
		    .series("PositivePoints", Plot.data().xy(positivex,positivey),
		        Plot.seriesOpts().line(Line.NONE).
		            marker(Plot.Marker.CIRCLE).
		            markerColor(Color.GREEN))
		    .series("NegativePoints", Plot.data().xy(negativex,negativey),
			        Plot.seriesOpts().line(Line.NONE).
			            marker(Plot.Marker.CIRCLE).
			            markerColor(Color.RED))
		    .series("last weights", Plot.data().xy(new double[]{min,max},new double[]{function(weights.get(weights.size()-1),new double[] {1,min,min}),function(weights.get(weights.size()-1),new double[] {1,max,max})}),
			        Plot.seriesOpts().line(Line.SOLID).
			        color(Color.BLUE).
			            marker(Plot.Marker.NONE))
		    .series("first weights", Plot.data().xy(new double[]{min,max},new double[]{function(weights.get(0),new double[] {1,min,min}),function(weights.get(0),new double[] {1,max,max})}),
			        Plot.seriesOpts().line(Line.SOLID).
			        	color(Color.MAGENTA).
			            marker(Plot.Marker.NONE))
		    .series("GRIDX", Plot.data().xy(new double[]{min,max},new double[]{0,0}),
			        Plot.seriesOpts().
			        line(Line.DASHED).
			        color(Color.BLACK).
			            marker(Plot.Marker.NONE))
		    .series("GRIDY", Plot.data().xy(new double[]{0,0},new double[]{min,max}),
			        Plot.seriesOpts().
			        line(Line.DASHED).
			        color(Color.BLACK).
			        	marker(Plot.Marker.NONE))
		    ;

		try {
			plot.save("res/"+name, "png");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
