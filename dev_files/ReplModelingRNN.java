package MNIST_DNN.dev_files; /** why are the losses Nans?  **/
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class ReplModelingRNN {
    public static ArrayList<ArrayList<Double>> x_train = new ArrayList<ArrayList<Double>>();
    public static ArrayList<Double> y_train = new ArrayList<Double>();
    public static int n_features = 10;
    public static double wH = 0.0;
    public static double wY = 0.0;
    public static double wX = 0.0;
    public static int numData = 60000;
    public static int numCols = 784;
    public static int numEpochs = 10000 ;
    public static double learningRate = 4;
    public static ArrayList<Double> nonActivatedOutputs = new ArrayList<Double>();
    public static ArrayList<Double> states = new ArrayList<Double>();
    public static double sigmoid(double x) {
        return 1/ (1 + Math.exp(-x));
    }

    public static double reLU(double x) {
        double rV = 0;
        if (x > 0) {rV = x;}
        return rV;
    }

    public static double tanh(double x) {
        return 2/(1 + Math.exp(-2 * x)) - 1;
    }


    //to save time later
    public static ArrayList<Double> flatten2D (ArrayList<ArrayList<Double>> x) {
        ArrayList<Double> flattenedStuff = new ArrayList<Double>();
        for (int row = 0; row < x.size(); row++) {
            for (int col = 0; col < x.get(row).size(); col++) {
                flattenedStuff.add(x.get(row).get(col));
            }
        }

        return flattenedStuff;
    }

    public static HashMap<String, Object> getData(int rows, int cols) {
        int x_axis = 0;
        ArrayList<ArrayList<Double>> x_train = new ArrayList<ArrayList<Double>>();
        ArrayList<Double> y_train = new ArrayList<Double>();
        for (int row = 0; row < rows; row++ ) {
            ArrayList<Double> currentData = new ArrayList<Double>();
            for (int col = 0; col < cols; col++) {
                currentData.add(Math.sin(x_axis)  -  x_axis * Math.cos(x_axis + reLU(x_axis) + sigmoid(x_axis)));
                x_axis += 1;
            }
            x_train.add(currentData);
            double potentialLabel = Math.sin(x_axis) -  x_axis * Math.cos(x_axis + reLU(x_axis) + sigmoid(x_axis));
            if (Math.round(potentialLabel) > 0) {
                y_train.add(1.0) ;
            }
            if (Math.round(potentialLabel) == 0) {
                y_train.add(0.0);
            }
            if (Math.round(potentialLabel) < 0) {
                y_train.add(-1.0);
            }
            x_axis += 1;
        }

        HashMap<String, Object> stuff = new HashMap<String, Object>();
        stuff.put("x_train", (Object) x_train);
        stuff.put("y_train", (Object) y_train);
        return stuff;
    }


    //main method
    public static void main(String[]args) throws Exception{
        Random rnd = new Random();
        HashMap<String, Object> allData = getData(numData, numCols);
        x_train = (ArrayList<ArrayList<Double>>) allData.get("x_train");
        y_train = (ArrayList<Double>) allData.get("y_train");
        System.out.println(x_train);
        System.out.println(y_train);
        weightInit();
        ArrayList<Double> outputs = forwardPropagation(x_train, wX, wY, wH);

        System.out.println("Wh : " + wH + "  Wy : "  + wY  + " Wx :" + wX);
        System.out.println("outputs : " + outputs);
        System.out.println("size : " + outputs.size());
        HashMap<String, Double> grads = backpropagation(wH, wX, wY, outputs, y_train);
        System.out.println("gradients : " + grads);
        System.out.println("losses : " + categoricalCrossentropy(outputs, y_train));

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            outputs = forwardPropagation(x_train, wX, wY, wH);
            grads = backpropagation(wH, wX, wY, outputs, y_train);
            System.out.println("gradients : " + grads);
            System.out.println("losses : " + categoricalCrossentropy(outputs, y_train));

            ArrayList<Double> weights = new ArrayList<Double>();
            ///dubm mistake I didn't give it anything for the stuff
            weights.add(wX);
            weights.add(wY);
            weights.add(wH);
            System.out.println(standardDeviation(weights));
            System.out.println(variance(weights));

            double wx_grad = grads.get("Wx_grad");
            double wy_grad = grads.get("Wy_grad");
            double wh_grad = grads.get("Wh_grad");
            wX -= learningRate * epoch *  wx_grad;
            wH -= learningRate * epoch * wh_grad;
            wY -= learningRate * epoch * wy_grad;
        }

        System.out.println("outputs : " + outputs);
        System.out.println("labels :  " + y_train);

        for (int z = 0 ; z < y_train.size(); z++) {System.out.println("output : " + outputs.get(z) + " label : " + y_train.get(z)); }

        int amountCorrect = 0;
        for (int z = 0; z < y_train.size(); z++) {
            if (Math.round(y_train.get(z)) == Math.round(outputs.get(z))) {amountCorrect += 1;}
        }
        System.out.println(amountCorrect);

        ArrayList<Double> predictStuff = new ArrayList<Double>();
        //-100.314661732
        //$$=	$$36.7183403154
        //$$=	$$72.316210972
        //$$=	$$-96.3037716352
        //$$=	$$8.41328295758
        //$$=	$$90.4983876228
        //$$=	$$-86.484070780

        predictStuff.add(-100.32);
        predictStuff.add(36.72);
        predictStuff.add(72.32);
        predictStuff.add(-96.30);
        predictStuff.add(8.41);
        predictStuff.add(90.5);
        predictStuff.add(-86.48);
        double prediction = predict(predictStuff);
        System.out.println("prediction : "  + prediction);
        System.out.println("Wx : " + wX + " Wy : " + wY + " Wh : " + wH);
        ArrayList<Double> weights = new ArrayList<Double>();
        weights.add(wX);
        weights.add(wY);
        weights.add(wH);
        System.out.println("STD : " + standardDeviation(weights));
        System.out.println("VAR : " + variance(weights));
    }

    public static int testandtrain() {
        Random rnd = new Random();
        HashMap<String, Object> allData = getData(numData, numCols);
        x_train = (ArrayList<ArrayList<Double>>) allData.get("x_train");
        y_train = (ArrayList<Double>) allData.get("y_train");
        System.out.println(x_train);
        System.out.println(y_train);
        weightInit();
        ArrayList<Double> outputs = forwardPropagation(x_train, wX, wY, wH);

        System.out.println("Wh : " + wH + "  Wy : "  + wY  + " Wx :" + wX);
        System.out.println("outputs : " + outputs);
        System.out.println("size : " + outputs.size());
        HashMap<String, Double> grads = backpropagation(wH, wX, wY, outputs, y_train);
        System.out.println("gradients : " + grads);
        System.out.println("losses : " + categoricalCrossentropy(outputs, y_train));

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            outputs = forwardPropagation(x_train, wX, wY, wH);
            grads = backpropagation(wH, wX, wY, outputs, y_train);
            System.out.println("gradients : " + grads);
            System.out.println("losses : " + categoricalCrossentropy(outputs, y_train));

            ArrayList<Double> weights = new ArrayList<Double>();
            ///dubm mistake I didn't give it anything for the stuff
            weights.add(wX);
            weights.add(wY);
            weights.add(wH);
            System.out.println(standardDeviation(weights));


            double wx_grad = grads.get("Wx_grad");
            double wy_grad = grads.get("Wy_grad");
            double wh_grad = grads.get("Wh_grad");
            wX -= learningRate * epoch *  wx_grad;
            wH -= learningRate * epoch * wh_grad;
            wY -= learningRate * epoch * wy_grad;
        }

        System.out.println("outputs : " + outputs);
        System.out.println("labels :  " + y_train);

        for (int z = 0 ; z < y_train.size(); z++) {System.out.println("output : " + outputs.get(z) + " label : " + y_train.get(z)); }

        int amountCorrect = 0;
        for (int z = 0; z < y_train.size(); z++) {
            if (Math.round(y_train.get(z)) == Math.round(outputs.get(z))) {amountCorrect += 1;}
        }

        return 200 - amountCorrect;
    }


    public static double standardDeviation(ArrayList<Double> x){
        double sum = 0;
        for(int i = 0; i < x.size(); i++){
            sum += (x.get(i) - average(x)) * (x.get(i) - average(x));
        }
        return Math.sqrt( sum / (x.size()) );
    }

    public static double variance(ArrayList<Double> x) {return standardDeviation(x) * standardDeviation(x);}

    public static void normalDistribution(double Wx, double Wy, double Wh) {
        //wait so the standard deviation and mu is of the Wx, Wy, Wh?

        double mean = (Wx + Wy + Wh)/3;
        ArrayList<Double> weights = new ArrayList<Double>();
        weights.add(wX);
        weights.add(wY);
        weights.add(wH);
        double std = standardDeviation(weights);
        wX = 1/( (Math.sqrt(2 * Math.PI) * std) * Math.exp(-1 * ((wX-mean) * (wX - mean))/(2 * std * std)) ); //looks correct?
        wY = 1/( (Math.sqrt(2 * Math.PI) * std) * Math.exp(-1 * ((wY-mean) * (wY - mean))/(2 * std * std)) );
        wH = 1/( (Math.sqrt(2 * Math.PI) * std) * Math.exp(-1 * ((wH-mean) * (wH - mean))/(2 * std * std)) );
    }





    //suprisingly repl does better than intellij, yee 80% accuracy, -1 is correct this is gooat thanks!!!!!!
    //hyperparam lemme check
    // wait but why is it still getting the sign wrong for some of them




    //weight init
    public static void weightInit() {
        //xavier  is 2/(fan_in + fan_out)
        //update wX, wY, wH
        Random rnd = new Random();
        wX = rnd.nextDouble();
        wY = rnd.nextDouble();
        wH = rnd.nextDouble();
        normalDistribution(wX, wY, wH);

    }

    //forward Propagation
    public static ArrayList<Double> forwardPropagation(ArrayList<ArrayList<Double>> inputs, double wX, double  wY, double wH) {

        ArrayList<Double> outputs = new ArrayList<Double>();
        for (int input = 0; input < inputs.size(); input++) {
            //we need to go through the run cycle for every single input inside of the input
            ArrayList<Double> currentInput = inputs.get(input);
            states = new ArrayList<Double>();
            states.add(wH);
            for (int time = 0; time < currentInput.size(); time++) {
                states.add(currentInput.get(time) * wX + lastInput(states) * wH);
            }
            //System.out.println("This is states : " + states + " size : " + states.size());
            nonActivatedOutputs.add(lastInput(states) * wY);
            outputs.add(tanh(lastInput(states) * wY));
        }


        return outputs;

    }

    //this function and the one on 188 are useless
    public static ArrayList<Double> elementWiseMult(ArrayList<Double> x, ArrayList<Double> y) {
        ArrayList<Double> w = new ArrayList<Double>();
        for (int i =0 ; i < x.size(); i++) {
            w.add(x.get(i) * y.get(i));
        }
        return w;

    }

    public static ArrayList<Double> elementWiseAdd(ArrayList<Double> x, ArrayList<Double> y) {
        ArrayList<Double> w = new ArrayList<Double>();
        for (int i =0 ; i < x.size(); i++) {
            w.add(x.get(i) + y.get(i));
        }
        return w;

    }

    public static double lastInput(ArrayList<Double> x) {
        return x.get(x.size() - 1);
    }


    public static ArrayList<Double> derivativeLosses(ArrayList<Double> outputs, ArrayList<Double> labels) {
        ArrayList<Double> losses = new ArrayList<Double>();
        for (int i = 0; i < outputs.size(); i++) {
            losses.add(-1 * labels.get(i)/outputs.get(i));
        }
        return losses;
    }

    public static double average(ArrayList<Double> x) {
        double sum =0;
        for (int a = 0; a < x.size(); a++) {
            sum += x.get(a);
        }
        return sum/x.size();
    }

    public static double tanhDeriv(double x) {
        return 1 - tanh(x) * tanh(x) ;
    }

    public static HashMap<String, Double> backpropagation(double wH, double wX, double wY, ArrayList<Double> outputs, ArrayList<Double> labels) {
        ArrayList<Double> derivLosses = derivativeLosses(outputs, labels);
        double hiddenStateWeightIter = Math.pow(wH, states.size() - 2); //todo check this!
        ArrayList<Double> dLdWXs = new ArrayList<Double>();
        for (int input = 0; input < outputs.size(); input++) {
            dLdWXs.add(derivLosses.get(input) * tanhDeriv(nonActivatedOutputs.get(input)) *  wY * hiddenStateWeightIter * x_train.get(input).get(0)); //todo is it zero?
        }
        double dLdWx = average(dLdWXs) * dLdWXs.size();

        //checked, check for averaging
        ArrayList<Double> dLdWYs = new ArrayList<Double>();
        for (int input = 0; input < outputs.size(); input++) {
            dLdWYs.add(derivLosses.get(input) *  tanhDeriv(nonActivatedOutputs.get(input)) * lastInput(states));
        }
        double dLdWY = average(dLdWYs) * dLdWYs.size();

        ArrayList<Double> dLdWHs = new ArrayList<Double>();
        for (int input = 0; input < outputs.size(); input++) {
            dLdWHs.add(derivLosses.get(input) *  tanhDeriv(nonActivatedOutputs.get(input)) * wY * hiddenStateWeightIter * wH);
        }

        double dLdWH = average(dLdWHs) * dLdWHs.size();

        HashMap<String, Double> gradients = new HashMap<String, Double>(); // {"Wx_grad" : whateverhellvalue, "Wy_grad" : whatever value}
        gradients.put("Wx_grad", dLdWx);
        gradients.put("Wh_grad", dLdWH);
        gradients.put("Wy_grad", dLdWY);
        return gradients;
    }

    public static double categoricalCrossentropy(ArrayList<Double>outputs, ArrayList<Double> labels) {
        double sum =0;
        for (int i =0; i < outputs.size(); i++) {
            sum += labels.get(i) * Math.log(outputs.get(i));
        }
        return sum * -1;
    }

    public static double predict(ArrayList<Double> input) {
        ArrayList<Double> predictedStates = new ArrayList<Double>();
        predictedStates.add(wH);
        for (int time = 0; time < input.size(); time++) {
            predictedStates.add(input.get(time) * wX + lastInput(predictedStates) * wH);
        }
        return tanh(lastInput(predictedStates) * wY);
    }






}
