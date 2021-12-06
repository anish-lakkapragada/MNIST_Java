package MNIST_DNN.dev_files;/*
Assumptions :
   - minus despite softmax for loss***/

import java.util.ArrayList;
import java.io.*;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

public class ScaledSigmoidMNIST  {
    public static ArrayList<ArrayList<Double>> x_train = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> y_train = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> x_test = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> y_test = new ArrayList<ArrayList<Double>>();
    //increase data, incerase neurons, don't increase features
    public static int numData = 120; //forward and back pass would hapepn quick iwth 500, 100, 32, 16, 50, 0.0004
    public static int numVal = 75;
    public static int numNeurons1 = 32;
    public static int numNeurons2 = 16;
    public static int numNeurons3 = 10;
    public static int numFeatures = 784;
    public static double learningRate = 0.005; //working 0.0001 with 200, 4, 2, 100, 1000 config
    public static double biasLearningRate = 0.002 ;
    public static double leaking = 0.0001;
    public static boolean bias = true;
    public static double scaleWeights = 1.0;
    public static int numEpochs = 10000;
    public static double e = 0.001;
    public static double gradClip = 1000;
    public static boolean randomState = true;
    public static ArrayList<ArrayList<Double>> weights1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> weights2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> weights3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> velocity1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> velocity2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> velocity3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> biasVelocity1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> biasVelocity2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> biasVelocity3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<Double> biases1 = new ArrayList<Double>();
    public static ArrayList<Double> biases2 = new ArrayList<Double>();
    public static ArrayList<Double> biases3 = new ArrayList<Double>();
    public static double momentum = 0;
    public static double biasMomentum = 0;
    public static boolean nesterov = false;
    public static ArrayList<ArrayList<Double>> pastCosts = new ArrayList<ArrayList<Double>>();
    public static ArrayList<Double> classes = new ArrayList<Double>();
    public static void getTrainData() throws Exception {
        String train_path = "/Users/anish/Java Fun/ML Java/src/mnist_train.csv";
        String line = "";
        BufferedReader br = new BufferedReader(new FileReader(train_path));
        int z = numData;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            ArrayList<Double> currentX_train = new ArrayList<Double>();
            for (int i = 0; i < values.length; i++) {
                if (i == 0) {y_train.add(oneHot(Double.valueOf(values[0])));}
                else {
                    currentX_train.add((Double.valueOf(values[i]))/255.0);
                }
            }
            x_train.add(currentX_train);
            z -= 1;
            if (z == 0) {break;}
        }
    }

    public static void getValidationData() throws Exception {
        String train_path = "/Users/anish/Java Fun/ML Java/src/mnist_test.csv";
        String line = "";
        BufferedReader br = new BufferedReader(new FileReader(train_path));
        int z = numVal;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            ArrayList<Double> currentX_train = new ArrayList<Double>();
            for (int i = 0; i < values.length; i++) {
                if (i == 0) {y_test.add(oneHot(Double.valueOf(values[0])));}
                else {
                    currentX_train.add((Double.valueOf(values[i]))/255.0);
                }
            }
            x_test.add(currentX_train);
            z -= 1;
            if (z == 0) {break;}
        }
    }

    public static ArrayList<Double> oneHot(double x) {
        ArrayList<Double> y = new ArrayList<Double>();
        for (int i = 0; i < 10; i++) {
            if (i == x) {y.add(1.0);}
            else {y.add(0.0);}
        }
        return y;
    }

    public static void setClasses() {
        classes.add(0.0);
        classes.add(1.0);
        classes.add(2.0);
        classes.add(3.0);
        classes.add(4.0);
        classes.add(5.0);
        classes.add(6.0);
        classes.add(7.0);
        classes.add(8.0);
        classes.add(9.0);
    }

    public static void weightInit() {
        Random rnd ;
        if (randomState) {rnd = new Random(9098);} else {rnd = new Random();}
        rnd.setSeed(rnd.nextInt(100));
        for (int feature = 0; feature < x_train.get(0).size(); feature++) {
            ArrayList<Double> featureToLayer = new ArrayList<Double>();
            ArrayList<Double> v1 = new ArrayList<Double>();
            for (int n1 = 0; n1 < numNeurons1;n1++) {
                //featureToLayer.add(Math.random() * Math.sqrt(2/(784 + numNeurons1))); w//todo weight init
                featureToLayer.add(rnd.nextGaussian());
                v1.add(0.0);
            }
            weights1.add(featureToLayer);
            System.out.println("W1 VAR : " + DimensionalityReduction.variance(featureToLayer));
            velocity1.add(v1);
        }

        rnd = new Random(rnd.nextInt(100));
        for (int feature = 0; feature < numNeurons1; feature++) {
            ArrayList<Double> featureToLayer = new ArrayList<Double>();
            ArrayList<Double> v2 = new ArrayList<Double>();
            for (int n1 = 0; n1 < numNeurons2;n1++) {
                featureToLayer.add(rnd.nextGaussian()); //* Math.sqrt(2/(numNeurons1 + numNeurons2))); //todo weight init
                v2.add(0.0);
            }
            weights2.add(featureToLayer);
            velocity2.add(v2);
        }

        rnd = new Random(rnd.nextInt(100));
        for (int feature = 0; feature < numNeurons2; feature++) {
            ArrayList<Double> featureToLayer = new ArrayList<Double>();
            ArrayList<Double> v2 = new ArrayList<Double>();
            for (int n1 = 0; n1 < numNeurons3;n1++) {
                featureToLayer.add(rnd.nextGaussian()); //* Math.sqrt(2/(numNeurons3 + numNeurons2))); //todo weight init
                v2.add(0.0);
            }
            weights3.add(featureToLayer);
            System.out.println("W3 VAR : " + DimensionalityReduction.variance(featureToLayer));
            velocity3.add(v2);
        }

    }

    public static void Xavier() {
        for (int feature = 0; feature < weights1.size(); feature++) {
            double fan_avg = (double) (784 + numNeurons1)/2.0;
            for (int col = 0; col < weights1.get(feature).size(); col++) {
                weights1.get(feature).set(col, weights1.get(feature).get(col) * Math.sqrt(1/fan_avg));
            }
        }

        for (int feature = 0; feature < weights2.size(); feature++) {
            double fan_avg = (double) (numNeurons1 + numNeurons2)/2.0;
            for (int col = 0; col < weights2.get(feature).size(); col++) {
                weights2.get(feature).set(col, weights2.get(feature).get(col) * Math.sqrt(1/fan_avg));
            }
        }

        for (int feature = 0; feature < weights3.size(); feature++) {
            double fan_avg = (double) (numNeurons2 + numNeurons3)/2.0;
            for (int col = 0; col < weights3.get(feature).size(); col++) {
                weights3.get(feature).set(col, weights3.get(feature).get(col) * Math.sqrt(1/fan_avg));
            }
        }
    }

    public static void biasInit() {
        Random rnd ;
        if (randomState) {rnd = new Random(232);} else {rnd = new Random();}
        if (bias) {
            /*for (int n1 = 0; n1 < numNeurons1; n1++) {biases1.add(rnd.nextDouble());}
            for (int n2 = 0; n2 < numNeurons2; n2++) {biases2.add(rnd.nextDouble());}
            for (int n3 = 0; n3 < numNeurons3; n3++) {biases3.add(rnd.nextDouble());}*/

            //xavier

            for (int n1 = 0; n1 < numNeurons1; n1++) {biases1.add(0.0);}
            for (int n2 = 0; n2 < numNeurons2; n2++) {biases2.add(0.0);}
            for (int n3 = 0; n3 < numNeurons3; n3++) {biases3.add(0.0);}
        }

        else {
            for (int n1 = 0; n1 < numNeurons1; n1++) {biases1.add(0.0);}
            for (int n2 = 0; n2 < numNeurons2; n2++) {biases2.add(0.0);}
            for (int n3 = 0; n3 < numNeurons3; n3++) {biases3.add(0.0);}
        }
    }

    public static void biasClear() {
        if(!bias) {
            for (int n1 = 0; n1 < numNeurons1; n1++) {biases1.set(n1, 0.0);}
            for (int n2 = 0; n2 < numNeurons2; n2++) {biases2.set(n2, 0.0);}
            for (int n3 = 0; n3 < numNeurons3; n3++) {biases3.add(0.0);}
        }
    }

    public static void biasMomentumInit() {
        Random rnd ;
        if (randomState) {rnd = new Random(127834);} else {rnd = new Random();}
        for (int i = 0; i < numData; i++) {
            ArrayList<Double> currentBiasV = new ArrayList<Double>();
            for (int n1 = 0; n1 < numNeurons1; n1++) {currentBiasV.add(rnd.nextDouble());}
            biasVelocity1.add(currentBiasV);
        }

        for (int i = 0; i < numData; i++) {
            ArrayList<Double> currentBiasV = new ArrayList<Double>();
            for (int n1 = 0; n1 < numNeurons2; n1++) {currentBiasV.add(rnd.nextDouble());}
            biasVelocity2.add(currentBiasV);
        }

        for (int i = 0; i < numData; i++) {
            ArrayList<Double> currentBiasV = new ArrayList<Double>();
            for (int n1 = 0; n1 < numNeurons3; n1++) {currentBiasV.add(rnd.nextDouble());}
            biasVelocity3.add(currentBiasV);
        }

    }


    public static ArrayList<ArrayList<Double>> transpose(ArrayList<ArrayList<Double>> x)  {
        ArrayList<ArrayList<Double>> transposedArray = new ArrayList<ArrayList<Double>>();
        //transpose start
        for (int i = 0 ; i < x.get(0).size(); i++) {
            ArrayList<Double>usable = new ArrayList<Double>();
            transposedArray.add(usable);
        }

        for (int point = 0; point < x.size(); point++) {
            for (int coordinate = 0; coordinate < x.get(point).size(); coordinate++) {
                //wriing dot product functinos and transpose functions in java is fun
                transposedArray.get(coordinate).add(x.get(point).get(coordinate));
            }
        }
        return (transposedArray);
    }

    public static ArrayList<ArrayList<Double>> reLU(ArrayList<ArrayList<Double>> x) {
        ArrayList<ArrayList<Double>> newX = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < x.size(); i++) {
            ArrayList<Double> current = new ArrayList<Double>();
            for (int j = 0; j < x.get(i).size(); j++) {
                current.add(1/(1 + Math.exp(-1 * x.get(i).get(j))));
            }
            newX.add(current);
        }
        return newX;
    }

    public static ArrayList<Double> vectorDivision(ArrayList<Double> x, ArrayList<Double>y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i = 0; i < x.size(); i++) {
            z.add(x.get(i)/y.get(i));
        }
        return z;
    }

    public static ArrayList<Double> vectorMultiply(ArrayList<Double> x, ArrayList<Double>y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i = 0; i < x.size(); i++) {
            z.add(x.get(i) * y.get(i));
        }
        return z;
    }

    public static double categoricalCrossentropy(ArrayList<Double>outputs, ArrayList<Double> labels) {
        double sum =0;
        for (int i =0; i < outputs.size(); i++) {
            sum += labels.get(i) * Math.log(outputs.get(i));
        }
        return sum * -1;
    }

    public static ArrayList<ArrayList<Double>> derivativeLosses(ArrayList<ArrayList<Double>> outputs, ArrayList<ArrayList<Double>> labels) {
        ArrayList<ArrayList<Double>> losses = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < outputs.size(); i++) {
            losses.add(scalarDivide(vectorDivision(labels.get(i), outputs.get(i)), -1));
        }
        return losses;
    }

    public static double softmax(double output, ArrayList<Double> outputs) {
        double den = 0;
        for (int i = 0; i < outputs.size(); i++) {
            den += Math.exp(outputs.get(i));
        }

        return Math.exp(output)/den;
    }

    public static ArrayList<Double> softmaxDerivative(ArrayList<Double> i, ArrayList<Double> j, boolean same) {
        ArrayList<Double> deriv = new ArrayList<Double>();
        if (same) {
            ArrayList<Double> newJ = new ArrayList<Double>();
            for (int z = 0; z < j.size(); z++) {newJ.add(1 - j.get(z));}
            deriv = vectorMultiply(i, newJ);
        }

        else {
            deriv =  scalarDivide(vectorMultiply(i ,j), -1);
        }
        return deriv;
    }

    public static double dotSum(ArrayList<Double> x, ArrayList<Double> y) {
        double sum =0;
        for (int i = 0; i < x.size(); i++) {sum += x.get(i) * y.get(i);}
        return sum;
    }

    public static HashMap<String, Object> forwardPropagation(ArrayList<ArrayList<Double>> inputs, ArrayList<ArrayList<Double>> weights1, ArrayList<ArrayList<Double>> weights2, ArrayList<ArrayList<Double>> weights3) throws Exception{
        /*Thread fp = new Thread();
        fp.start();*/

        ArrayList<ArrayList<Double>> activation1 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> activation2 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> activation3 = new ArrayList<ArrayList<Double>>();
        //activation1
        for (int input = 0; input < inputs.size(); input++) {
            ArrayList<Double> currentActivation = new ArrayList<Double>();
            for (int neuron = 0; neuron < transpose(weights1).size(); neuron++) {
                /*(double sum = 0;
                for (int feature = 0; feature < numFeatures; feature++) {
                    sum += transpose(weights1).get(neuron).get(feature) * x_train.get(input).get(feature);
                }*/

                currentActivation.add(dotSum(transpose(weights1).get(neuron), x_train.get(input)));
            }
            activation1.add(currentActivation);
            System.out.println("Activation 1 : " + input);
        }
        activation1 = reLU(biasAdd(activation1, biases1));

        //activation2
        for (int input = 0; input < inputs.size(); input++) {
            ArrayList<Double> currentActivation = new ArrayList<Double>();
            for (int neuron = 0; neuron < transpose(weights2).size(); neuron++) {
                /*double sum = 0;
                for (int feature = 0; feature < numNeurons1; feature++) {
                    sum += transpose(weights2).get(neuron).get(feature) * activation1.get(input).get(feature);
                }*/
                currentActivation.add(dotSum(transpose(weights2).get(neuron), activation1.get(input)));
            }
            System.out.println("Activation 2 : " + input);
            activation2.add(currentActivation);
        }

        activation2 = reLU(biasAdd(activation2, biases2));

        //activation3 aka da predictions - need to go through softmax
        for (int input = 0; input < inputs.size(); input++) {
            ArrayList<Double> currentActivation = new ArrayList<Double>();
            for (int neuron = 0; neuron < transpose(weights3).size(); neuron++) {
                /*double sum = 0;
                for (int feature = 0; feature < numNeurons1; feature++) {
                    sum += transpose(weights2).get(neuron).get(feature) * activation1.get(input).get(feature);
                }*/
                currentActivation.add(dotSum(transpose(weights3).get(neuron), activation2.get(input)));
            }
            System.out.println("Activation 3 : " + input);
            activation3.add(currentActivation);
        }

        activation3 = reLU(biasAdd(activation3, biases3)); //hasn't gone through softmax

        System.out.println("implementing softmax");
        /*softmax time
        for (int pred  = 0; pred < activation3.size(); pred++) {
            for (int output = 0; output < activation3.get(pred).size(); output++) {
                activation3.get(pred).set(output, softmax(activation3.get(pred).get(output), activation3.get(pred)));
            }
        }*/

        ArrayList<ArrayList<Double>> softmaxA3 = softmaxActivation(activation3);
        System.out.println("Done implementation");
        HashMap<String, Object> results = new HashMap<String, Object>();
        results.put("activation3", (Object) softmaxA3);
        results.put("rawA3", (Object) activation3);
        results.put("activation2", (Object) activation2);
        results.put("activation1", (Object) activation1);
        return results;
    }

    public static ArrayList<ArrayList<Double>> softmaxActivation(ArrayList<ArrayList<Double>> activation) {
        /*Thread softmaxThread = new Thread();
        softmaxThread.start();*/
        ArrayList<ArrayList<Double>> newActivation = new ArrayList<ArrayList<Double>>();
        ArrayList<Double> currentActiv = new ArrayList<Double>();
        for (int pred  = 0; pred < activation.size(); pred++) {
            currentActiv = new ArrayList<Double>();
            for (int output = 0; output < activation.get(pred).size(); output++) {
                currentActiv.add(softmax(activation.get(pred).get(output), activation.get(pred)));
            }
            newActivation.add(currentActiv);
        }
        return newActivation;
    }

    public static ArrayList<Double> mostCommonLabels (ArrayList<ArrayList<Double>> a3) {
        ArrayList<Double> newPred = new ArrayList<Double>();
        for (int pred = 0; pred < a3.size(); pred++) {
            double highest_pred = 0;
            for (int output = 0; output < a3.get(pred).size(); output++) {
                if (highest_pred < a3.get(pred).get(output)) {highest_pred = a3.get(pred).get(output);}
            }
            newPred.add((double) a3.get(pred).indexOf(highest_pred));
        }
        return newPred;
    }


    public static void nesterovAccel() {
        //use nesterov's trick if needed
        if (nesterov) {

            for (int feature = 0; feature < weights1.size(); feature++){
                ArrayList<Double> currentList = new ArrayList<Double>();
                for (int neuron = 0; neuron < weights1.get(feature).size(); neuron++) {
                    currentList.add(weights1.get(feature).get(neuron)  -  momentum * velocity1.get(feature).get(neuron));
                }
                weights1.set(feature, currentList);
            }


            for (int feature = 0; feature < weights2.size(); feature++){
                ArrayList<Double> currentList = new ArrayList<Double>();
                for (int neuron = 0; neuron < weights2.get(feature).size(); neuron++) {
                    currentList.add(weights2.get(feature).get(neuron)  -  momentum * velocity2.get(feature).get(neuron));
                }
                weights2.set(feature, currentList);
            }


            for (int feature = 0; feature < weights3.size(); feature++){
                ArrayList<Double> currentList = new ArrayList<Double>();
                for (int neuron = 0; neuron < weights3.get(feature).size(); neuron++) {
                    currentList.add(weights3.get(feature).get(neuron)  -  momentum * velocity3.get(feature).get(neuron));
                }
                weights3.set(feature, currentList);
            }
        }
    }

    public static ArrayList<ArrayList<Double>> gradientClipping(ArrayList<ArrayList<Double>> gradients) {
        ArrayList<ArrayList<Double>> newGrads = gradients;
        for (int i = 0; i < gradients.size(); i++) {
            for (int j = 0; j < gradients.get(i).size(); j++) {
                if (gradients.get(i).get(j) > gradClip) {
                    newGrads.get(i).set(j, gradClip);
                }
            }
        }

        return newGrads;
    }

    /*public static double Xavier(ArrayList<Double> x, int fan_in, int fan_out) {
        double std = standardDeviation(x);
        double firstTerm = 1/(std * Math.sqrt(Math.PI * 2));

    }*/

    public static ArrayList<Double> scalarAdd(ArrayList<Double> x, double y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i = 0; i < x.size(); i++) {z.add(x.get(i) + y);}
        return z;
    }

    public static ArrayList<Double> scalarMultiply(ArrayList<Double> x, double y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i = 0; i < x.size(); i++) {z.add(x.get(i) * y);}
        return z;
    }

    public static HashMap<String, Object> backPropagation(ArrayList<ArrayList<Double>> outputs, ArrayList<ArrayList<Double>> labels, ArrayList<ArrayList<Double>> rawOutputs, ArrayList<ArrayList<Double>> activation2, ArrayList<ArrayList<Double>> activation1) throws Exception {
        nesterovAccel();
        //first thing is to calculate errors
        ArrayList<ArrayList<Double>> dLdYh = new ArrayList<ArrayList<Double>>();
        for (int i = 0 ; i < outputs.size(); i++) {
            //dLdYh.add(scalarDivide(vectorSubtract(outputs.get(i), labels.get(i)), labels.size()));
            dLdYh.add(scalarMultiply(vectorDivision(labels.get(i), scalarAdd(outputs.get(i), e)), -1));
        }

        System.out.println("Finished finding errors  : " + dLdYh);

        //next thing is to get the tensor in the shape len(d) * 10 * 10
        ArrayList<ArrayList<ArrayList<Double>>> dYhdZ3 = new ArrayList<ArrayList<ArrayList<Double>>>();
        for (int output = 0; output < rawOutputs.size(); output++) {
            ArrayList<ArrayList<Double>> jacobian = softmaxDeriv(rawOutputs.get(output));
            dYhdZ3.add(jacobian);
        }

        System.out.println("Finished dYhdZ3 ");

        //yay done with the tensor, time to multiply it into a matrix!
        ArrayList<ArrayList<Double>> dLdZ3 = new ArrayList<ArrayList<Double>>();
        for (int deriv = 0; deriv <  dYhdZ3.size(); deriv++) {
            //inside a 10 x 10 matrix for dYhdZ3.get(deriv)
            dLdZ3.add(dot(dYhdZ3.get(deriv), dLdYh.get(deriv)));
        }

        System.out.println("Finished dLdZ3 ");

        //time to multiply and create grads3!
        ArrayList<ArrayList<Double>> gradients3 = new ArrayList<ArrayList<Double>>();
        ArrayList<Double> currentGrad = new ArrayList<Double>();
        for (int row = 0; row < transpose(activation2).size(); row++) {
            /*for (int col = 0; col < dLdZ3.get(0).size(); col++) {
                currentGrad.add(dotSum(transpose(activation2).get(row), getCol(dLdZ3, col)));
            }
            gradients3.add(currentGrad);*/
            currentGrad = dot(transpose(dLdZ3), transpose(activation2).get(row));
            gradients3.add(currentGrad);
        }

        System.out.println("Finished grad3 ");

        ArrayList<ArrayList<Double>> dLdA2 = new ArrayList<ArrayList<Double>>();
        for (int row = 0; row < dLdZ3.size(); row++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int outputNeurons = 0; outputNeurons < weights3.size(); outputNeurons++) {
                currentDeriv.add(dotSum(dLdZ3.get(row), weights3.get(outputNeurons)));
            }
            dLdA2.add(currentDeriv);
        }

        System.out.println("Finished dLdA2 ");

        ArrayList<ArrayList<Double>> reluDerivZ2 = new ArrayList<ArrayList<Double>>();
        for (int row = 0; row < activation2.size(); row++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int col = 0; col < activation2.get(row).size(); col++) {
                currentDeriv.add(activation2.get(row).get(col) * (1 - activation2.get(row).get(col)));
            }
            reluDerivZ2.add(currentDeriv);
        }

        System.out.println("Finished reluDerivZ2 ");

        ArrayList<ArrayList<Double>> dLdZ2 = new ArrayList<ArrayList<Double>>();
        for (int row = 0; row < dLdA2.size(); row++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int col = 0; col < dLdA2.get(row).size(); col++) {
                currentDeriv.add(dLdA2.get(row).get(col) * reluDerivZ2.get(row).get(col));
            }
            dLdZ2.add(currentDeriv);
        }

        System.out.println("Finished dLdZ2 ");

        //todo clear mistake here, transposeget grads2
        ArrayList<ArrayList<Double>> gradients2 = new ArrayList<ArrayList<Double>>();
        currentGrad = new ArrayList<Double>();
        for (int row = 0; row < transpose(activation1).size(); row++) {
            /*for (int col = 0; col < dLdZ2.get(0).size(); col++) {
                currentGrad.add(dotSum(transpose(activation1).get(row), getCol(dLdZ2, col)));
            }
            gradients2.add(currentGrad);*/
            currentGrad = dot(transpose(dLdZ2), transpose(activation1).get(row));
            gradients2.add(currentGrad);
        }

        System.out.println("Finished grad2");


        //implement grads1
        ArrayList<ArrayList<Double>> dLdA1 = new ArrayList<ArrayList<Double>>();
        for (int row = 0; row < dLdZ2.size(); row++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int outputNeurons = 0; outputNeurons < weights2.size(); outputNeurons++) {
                currentDeriv.add(dotSum(dLdZ2.get(row), weights2.get(outputNeurons)));
            }
            dLdA1.add(currentDeriv);
        }

        System.out.println("Finished dLdA1 ");

        ArrayList<ArrayList<Double>> reluDerivZ1 = new ArrayList<ArrayList<Double>>();
        for (int row = 0; row < activation1.size(); row++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int col = 0; col < activation1.get(row).size(); col++) {
                currentDeriv.add(activation1.get(row).get(col) * (1 - activation1.get(row).get(col)));
            }
            reluDerivZ1.add(currentDeriv);
        }

        System.out.println("Finished reluderivZ1 ");

        ArrayList<ArrayList<Double>> dLdZ1 = new ArrayList<ArrayList<Double>>();
        for (int row = 0; row < reluDerivZ1.size(); row++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int col = 0; col < reluDerivZ1.get(0).size(); col++)  {
                currentDeriv.add(reluDerivZ1.get(row).get(col) * dLdA1.get(row).get(col));
            }
            dLdZ1.add(currentDeriv);
        }

        System.out.println("Finished dLdZ1 ");


        //TODO ALWAYS ZERO GRADS
        ArrayList<ArrayList<Double>> gradients1 = new ArrayList<ArrayList<Double>>();
        currentGrad = new ArrayList<Double>();
        for (int row = 0; row < transpose(x_train).size(); row++) {
            /*currentGrad = new ArrayList<Double>();
            for (int col = 0; col < dLdZ1.get(0).size(); col++) {
                currentGrad.add(dotSum(transpose(x_train).get(row), getCol(dLdZ1, col)));
            }*/
            currentGrad = dot(transpose(dLdZ1), transpose(x_train).get(row));
            //System.out.println("transpose x_train : " + transpose(x_train).get(row));
            gradients1.add(currentGrad);
        }

        System.out.println("Finished grad1 ");

        System.out.println("gradClip1");

        System.out.println("grad1 : " + gradients1);
        System.out.println("grad2 : " + gradients2);
        System.out.println("grad3  :" + gradients3);

        /*gradients1 = gradientClipping(gradients1);
        gradients2 = gradientClipping(gradients2);
        gradients3 = gradientClipping(gradients3);

        /*System.out.println("grad1 : " + gradients1.size() + " " + gradients1.get(0).size());
        System.out.println("grad2 : " + gradients2.size() + " " + gradients2.get(0).size());
        System.out.println("grad3 : " + gradients3.size() + " " + gradients3.get(0).size());
        System.out.println("Bias_grad3 : " + dLdZ3);
        System.out.println("Bias_grad2 : " + dLdZ2);
        System.out.println("Bias_grad1 : " + dLdZ1);*/

        HashMap<String, Object> gradients = new HashMap<String, Object>();
        gradients.put("grad1", (Object) gradients1);
        gradients.put("grad2", (Object) gradients2);
        gradients.put("grad3", (Object) gradients3);
        gradients.put("bias_grad3", (Object) dLdZ3);
        gradients.put("bias_grad2", (Object) dLdZ2);
        gradients.put("bias_grad1", (Object) dLdZ1);
        return gradients;
    }


    public static ArrayList<ArrayList<Double>> softmaxDeriv(ArrayList<Double> output) {
        ArrayList<ArrayList<Double>> jacobian = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < output.size(); i++) {
            ArrayList<Double> currentRow = new ArrayList<Double>();
            for (int j = 0 ; j < output.size(); j++) {
                if (i == j) {
                    currentRow.add(output.get(i) * (1 - output.get(j)));
                }
                else {
                    currentRow.add(-1 * output.get(i) * output.get(j));
                }
            }
            jacobian.add(currentRow);
        }
        return jacobian; //it better be square matrix
    }

    public static ArrayList<Double> dot(ArrayList<ArrayList<Double>> x, ArrayList<Double> y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int row = 0; row < x.size(); row++) {
            z.add(dotSum(x.get(row), y));
        }
        return z;

    }

    public static double vectorSum(ArrayList<Double> x ) {
        double sum = 0;
        for (int i = 0; i < x.size(); i++) {
            sum += x.get(i);
        }
        return sum;
    }

    public static ArrayList<Double> getCol(ArrayList<ArrayList<Double>> x, int col) {
        ArrayList<Double> column = new ArrayList<Double>();
        for (int i = 0; i < x.size(); i++) {column.add(x.get(i).get(col));}
        return column;
    }

    public static ArrayList<ArrayList<Double>> biasAdd(ArrayList<ArrayList<Double>> activation, ArrayList<Double> bias) {
        ArrayList<ArrayList<Double>> newActivation = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < activation.size(); i++) {
            newActivation.add(vectorAdd(bias, activation.get(i)));
        }
        return newActivation;
    }

    public static ArrayList<Double> biasUpdate(ArrayList<Double> bias, ArrayList<ArrayList<Double>> updates) {
        ArrayList<Double> newBias = bias;
        for (int col = 0; col < updates.get(0).size(); col++) {
            ArrayList<Double> currentCol = getCol(updates, col);
            for (int i = 0; i < currentCol.size(); i++) {
                newBias.set(col, newBias.get(col) - currentCol.get(i) * biasLearningRate);
            }
        }
        return newBias;
    }

    public static ArrayList<Double> vectorAdd(ArrayList<Double>a1, ArrayList<Double>a2) {
        //here a1 + a2
        ArrayList<Double> answer = new ArrayList<Double>();
        for (int dim = 0 ; dim < a1.size(); dim ++) {
            answer.add(a1.get(dim) + a2.get(dim));
        }

        return answer;
    }


    public static ArrayList<Double> vectorSubtract(ArrayList<Double>a1, ArrayList<Double>a2) {
        //here a1 + a2
        ArrayList<Double> answer = new ArrayList<Double>();
        for (int dim = 0 ; dim < a1.size(); dim ++) {
            answer.add(a1.get(dim) - a2.get(dim));
        }

        return answer;
    }



    public static void momentumGD(ArrayList<ArrayList<Double>> gradients1, ArrayList<ArrayList<Double>> gradients2, ArrayList<ArrayList<Double>> gradients3, ArrayList<ArrayList<Double>> biasGrad1, ArrayList<ArrayList<Double>> biasGrad2, ArrayList<ArrayList<Double>> biasGrad3) {
        ArrayList<ArrayList<Double>> newVelocityGrad1 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> newVelocityGrad2 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> newVelocityGrad3 = new ArrayList<ArrayList<Double>>();

        //standard momentum below

        for (int feature = 0; feature < gradients1.size(); feature++) {
            ArrayList<Double> currentList = new ArrayList<Double>();
            for (int neuron = 0; neuron < gradients1.get(feature).size(); neuron++) {
                //newVelocityGrad1.add(momentum * (double) velocity1.get(feature).get(neuron) + gradients1.get(feature).get(neuron));
                currentList.add(momentum * (double) velocity1.get(feature).get(neuron) + learningRate * gradients1.get(feature).get(neuron));
            }

            newVelocityGrad1.add(currentList);
        }

        for (int feature = 0; feature < gradients2.size(); feature++) {
            ArrayList<Double> currentList = new ArrayList<Double>();
            for (int neuron = 0; neuron < gradients2.get(feature).size(); neuron++) {
                //newVelocityGrad1.add(momentum * (double) velocity1.get(feature).get(neuron) + gradients1.get(feature).get(neuron));
                currentList.add(momentum * (double) velocity2.get(feature).get(neuron) + learningRate * gradients2.get(feature).get(neuron));
            }

            newVelocityGrad2.add(currentList);
        }

        for (int feature = 0; feature < gradients3.size(); feature++) {
            ArrayList<Double> currentList = new ArrayList<Double>();
            for (int neuron = 0; neuron < gradients3.get(feature).size(); neuron++) {
                //newVelocityGrad1.add(momentum * (double) velocity1.get(feature).get(neuron) + gradients1.get(feature).get(neuron));
                currentList.add(momentum * (double) velocity3.get(feature).get(neuron) + learningRate * gradients3.get(feature).get(neuron));
            }

            newVelocityGrad3.add(currentList);
        }

        velocity1 = newVelocityGrad1;
        velocity2 = newVelocityGrad2;
        velocity3 = newVelocityGrad3;

        //update weights
        for (int feature = 0; feature < weights1.size(); feature++) {
            for (int neuron = 0; neuron < weights1.get(feature).size(); neuron++) {
                weights1.get(feature).set(neuron, weights1.get(feature).get(neuron) - velocity1.get(feature).get(neuron));
            }
        }

        System.out.println("Applied w1 grad");

        for (int feature = 0; feature < weights2.size(); feature++) {
            for (int neuron = 0; neuron < weights2.get(feature).size(); neuron++) {
                weights2.get(feature).set(neuron, weights2.get(feature).get(neuron) - velocity2.get(feature).get(neuron));
            }
        }

        System.out.println("Applied w2 grad");

        for (int feature = 0; feature < weights3.size(); feature++) {
            for (int neuron = 0; neuron < weights3.get(feature).size(); neuron++) {
                weights3.get(feature).set(neuron, weights3.get(feature).get(neuron) - velocity3.get(feature).get(neuron));
            }
        }

        System.out.println("Applied w3 grad");

        if (bias) {
            ArrayList<ArrayList<Double>> newBiasVelocity1 = new ArrayList<ArrayList<Double>>();
            ArrayList<ArrayList<Double>> newBiasVelocity2 = new ArrayList<ArrayList<Double>>();
            ArrayList<ArrayList<Double>> newBiasVelocity3 = new ArrayList<ArrayList<Double>>();

            for (int row = 0; row < biasGrad1.size(); row++) {
                ArrayList<Double> currentList = new ArrayList<Double>();
                for (int col = 0; col < biasGrad1.get(row).size(); col++) {
                    currentList.add(biasMomentum * biasVelocity1.get(row).get(col) + biasLearningRate * biasGrad1.get(row).get(col));
                }
                newBiasVelocity1.add(currentList);
            }


            for (int row = 0; row < biasGrad1.size(); row++) {
                ArrayList<Double> currentList = new ArrayList<Double>();
                for (int col = 0; col < biasGrad1.get(row).size(); col++) {
                    currentList.add(biasMomentum * biasVelocity1.get(row).get(col) + biasLearningRate * biasGrad1.get(row).get(col));
                }
                newBiasVelocity1.add(currentList);
            }

            for (int row = 0; row < biasGrad3.size(); row++) {
                ArrayList<Double> currentList = new ArrayList<Double>();
                for (int col = 0; col < biasGrad3.get(row).size(); col++) {
                    currentList.add(biasMomentum * biasVelocity3.get(row).get(col) + biasLearningRate * biasGrad3.get(row).get(col));
                }
                newBiasVelocity3.add(currentList);
            }

            biasVelocity1 = newBiasVelocity1;
            biasVelocity2 = newBiasVelocity2;
            biasVelocity3 = newBiasVelocity3;

            for (int row = 0; row < biasVelocity3.size(); row++) {
                biases3 = vectorSubtract(biases3, scalarMultiply(biasVelocity3.get(row),  biasLearningRate));
            }

            for (int row = 0; row < biasVelocity2.size(); row++) {
                biases2 = vectorSubtract(biases2, scalarMultiply(biasVelocity2.get(row), biasLearningRate));
            }

            for (int row = 0; row < biasVelocity1.size(); row++) {
                biases1 = vectorSubtract(biases1, scalarMultiply(biasVelocity1.get(row), biasLearningRate));
            }
        }
    }

    public static ArrayList<Double> scalarDivide(ArrayList<Double> x, double y) {
        ArrayList<Double> newStuff = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {newStuff.add(x.get(i)/y);}
        return newStuff;
    }


    public static double vanillaEvaluation(ArrayList<Double> outputs, ArrayList<Double> labels) {
        int amountCorrect = 0;
        for (int i = 0; i < outputs.size(); i++) {
            if (Math.round(outputs.get(i)) == Math.round(labels.get(i))) {
                amountCorrect += 1;
            }
        }

        return amountCorrect;
    }


    public static void main(String[]args) throws Exception{
        getTrainData();
        getValidationData();
        setClasses();
        DimensionalityReduction preserver = new DimensionalityReduction();
        preserver.reduceDimensions();
        /*for (int d = 0; d < x_train.size(); d++) {
            System.out.println("X_train : " + x_train.get(d) + " label : " + y_train.get(d));
            System.out.println(x_train.get(d).size());
            //Thread.sleep(300);
        }
        System.out.println(x_train.size() + " " +  y_train.size());*/ //to check data correctly
        weightInit();
        biasInit();
        Xavier();
        biasMomentumInit();
        System.out.println("weight + biases init done!");

        /*System.out.println("weights1.size() : " + weights1.size() + " weights1.get(0).size() : " + weights1.get(0).size());
        System.out.println("weights2.size() : " + weights2.size() + "weights2.get(0).size() : " + weights2.get(0).size());
        System.out.println("weights3.size() : " + weights3.size());*/ //to check weights correctly

        System.out.println(weights1);
        System.out.println(weights2);
        System.out.println(weights3);
        System.out.println(biases1);
        System.out.println(biases2);
        System.out.println(biases3);

        System.out.println("VAR W1 : " + DimensionalityReduction.variance(ReplModelingRNN.flatten2D(weights1)));
        System.out.println("VAR W2 : " + DimensionalityReduction.variance(ReplModelingRNN.flatten2D(weights2)));
        System.out.println("VAR W3 : " + DimensionalityReduction.variance(ReplModelingRNN.flatten2D(weights3)));
        Thread.sleep(20000);

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            learningRate = learningRate * 2;
            biasLearningRate = biasLearningRate * 2;
            HashMap<String, Object> allForward = forwardPropagation(x_train, weights1, weights2, weights3);
            ArrayList<ArrayList<Double>> a3 = (ArrayList<ArrayList<Double>>) allForward.get("activation3");
            ArrayList<ArrayList<Double>> a1 = (ArrayList<ArrayList<Double>>) allForward.get("activation1");
            ArrayList<ArrayList<Double>> a2 = (ArrayList<ArrayList<Double>>) allForward.get("activation2");
            ArrayList<ArrayList<Double>> rawA3 = (ArrayList<ArrayList<Double>>) allForward.get("rawA3");
            HashMap<String, Object> gradients = backPropagation(a3, y_train, rawA3,  a2, a1);
            System.out.println("got all grads!");
            ArrayList<ArrayList<Double>> grad3 = (ArrayList<ArrayList<Double>>) gradients.get("grad3");
            ArrayList<ArrayList<Double>> grad2 = (ArrayList<ArrayList<Double>>) gradients.get("grad2");
            ArrayList<ArrayList<Double>> grad1 = (ArrayList<ArrayList<Double>>) gradients.get("grad1");
            ArrayList<ArrayList<Double>> bgrad3 = (ArrayList<ArrayList<Double>>) gradients.get("bias_grad3");
            ArrayList<ArrayList<Double>> bgrad2 = (ArrayList<ArrayList<Double>>) gradients.get("bias_grad2");
            ArrayList<ArrayList<Double>> bgrad1 = (ArrayList<ArrayList<Double>>) gradients.get("bias_grad1");
            HashMap<String, Object> testForward = forwardPropagation(x_test, weights1, weights2, weights3);
            ArrayList<ArrayList<Double>> testA3 = (ArrayList<ArrayList<Double>>) testForward.get("activation3");

            System.out.println("RAW3 : " + rawA3);
            for (int a = 0; a < rawA3.size(); a++) {System.out.println(rawA3.get(a));}
            System.out.println("grad1 : " + grad1);
            System.out.println("grad2 : " + grad2);
            System.out.println("grad3  :" + grad3);
            System.out.println("Vanilla train : " + vanillaEvaluation(mostCommonLabels(a3), mostCommonLabels(y_train)));
            System.out.println("Vanilla test : " + vanillaEvaluation(mostCommonLabels(testA3), mostCommonLabels(y_test)));
            System.out.println("Past costs : " + pastCosts);

            ArrayList<Double> currentCost = new ArrayList<Double>();
            currentCost.add(vanillaEvaluation(mostCommonLabels(a3), mostCommonLabels(y_train)));
            currentCost.add(vanillaEvaluation(mostCommonLabels(testA3), mostCommonLabels(y_test)));
            pastCosts.add(currentCost);

            if (momentum == 0 ) {
                double m = x_train.size();
                for (int row = 0; row < weights3.size(); row++) {
                    for (int col = 0; col < weights3.get(row).size(); col++) {
                        weights3.get(row).set(col, weights3.get(row).get(col) - (learningRate/m) * grad3.get(row).get(col));
                    }
                }

                for (int row = 0; row < weights2.size(); row++) {
                    for (int col = 0; col < weights2.get(row).size(); col++) {
                        weights2.get(row).set(col, weights2.get(row).get(col) - (learningRate/m) * grad2.get(row).get(col));
                    }
                }

                for (int row = 0; row < weights1.size(); row++) {
                    for (int col = 0; col < weights1.get(row).size(); col++) {
                        weights1.get(row).set(col, weights1.get(row).get(col) - (learningRate/m) * grad1.get(row).get(col));
                    }
                }


                if (bias) {
                    for (int row = 0; row < bgrad3.size(); row++) {
                        biases3 = vectorSubtract(biases3, scalarMultiply(bgrad3.get(row),  biasLearningRate/m));
                    }

                    for (int row = 0; row < bgrad2.size(); row++) {
                        biases2 = vectorSubtract(biases2, scalarMultiply(bgrad2.get(row), biasLearningRate/m));
                    }

                    for (int row = 0; row < bgrad1.size(); row++) {
                        biases1 = vectorSubtract(biases1, scalarMultiply(bgrad1.get(row), biasLearningRate/m));
                    }
                }
            }

            else {
                System.out.println("Gradients1 : " + grad1);
                System.out.println("Gradients2 : " + grad2);
                System.out.println("Gradients3 : " + grad3);
                momentumGD(grad1, grad2, grad3, bgrad1, bgrad2, bgrad3);
            }

            System.out.println("EPOCH : " + epoch);
            /*System.out.println("Activation 3 : " + a3);
            System.out.println("Gradients1 : " + grad1);
            System.out.println("Gradients2 : " + grad2);
            System.out.println("Gradients3 : " + grad3);
            System.out.println("biases1 : " + biases1);
            System.out.println("train A3 : " + mostCommonLabels(a3));
            System.out.println("train lbl:" + mostCommonLabels(y_train));
            System.out.println("This is y_train : " + y_train);

            System.out.println("test A3 : " + mostCommonLabels(testA3));
            System.out.println("test lbl :" + mostCommonLabels(y_test));

            System.out.println("Weights1 : " + weights1);
            System.out.println("Weights2 : " + weights2);
            System.out.println("Weights3 : " + weights3);
            System.out.println("Bias1 : " + biases1);
            System.out.println("Bias2 : " + biases2);
            System.out.println("Bias3 : " + biases3);*/

            Thread.sleep(7500);

            /*

            HashMap<String, Object> grads = backPropagation(outputs, y_train, a2, a1);
            System.out.println("got grads!");
            ArrayList<Double> grad3 = (ArrayList<Double>) grads.get("gradient3");
            ArrayList<ArrayList<Double>> grad2 = (ArrayList<ArrayList<Double>>) grads.get("gradient2");
            ArrayList<ArrayList<Double>> grad1 = (ArrayList<ArrayList<Double>>) grads.get("gradient1");
            ArrayList<Double> biasGrad3 = (ArrayList<Double>) grads.get("bias_grad3");
            ArrayList<ArrayList<Double>> biasGrad2 = (ArrayList<ArrayList<Double>>) grads.get("bias_grad2");
            ArrayList<ArrayList<Double>> biasGrad1 = (ArrayList<ArrayList<Double>>) grads.get("bias_grad1");
            System.out.println("grad3 : " + grad3);
            System.out.println("grad2 : " + grad2);



            System.out.println("EPOCH : " + epoch + " new cost : " + categoricalCrossentropy(outputs, y_train) + " val cost : " + vanillaEvaluation(y_pred, y_test) + " fancy CC : " + categoricalCrossentropy(y_pred, y_test));
            ArrayList<Double> costs = new ArrayList<Double>();
            costs.add(categoricalCrossentropy(outputs, y_train));
            costs.add(vanillaEvaluation(y_pred, y_test));
            pastCosts.add(costs);

            System.out.println("Y_pred : " +y_pred);
            System.out.println("Y_test : " + y_test);

            if(momentum > 0) {
                momentumGD(grad1, grad2, grad3, biasGrad1, biasGrad2, biasGrad3);
            }

            else {
                for (int z =0; z < weights3.size(); z++) {
                    weights3.set(z, weights3.get(z) - learningRate * grad3.get(z));
                }
                System.out.println("Applied w3 grad");

                for (int row = 0; row < weights2.size(); row++) {
                    for (int col = 0; col < weights2.get(row).size(); col++) {
                        weights2.get(row).set(col, weights2.get(row).get(col) - learningRate * grad2.get(row).get(col));
                    }
                }

                System.out.println("Applied w2 grad");
                for (int row = 0; row < weights1.size(); row++) {
                    for (int col = 0; col < weights1.get(row).size(); col++) {
                        weights1.get(row).set(col, weights1.get(row).get(col) - learningRate * grad1.get(row).get(col));
                    }
                }

                biases1 = biasUpdate(biases1, biasGrad1);
                biases2 = biasUpdate(biases2, biasGrad2);
                for (int i = 0 ; i < biasGrad3.size(); i++) {biases3 -= biasLearningRate * biasGrad3.get(i);}

                biasClear();
            }

            System.out.println("Applied w1 grad");

            System.out.println("weights1 : " + weights1);
            System.out.println("weights2 : " + weights2);
            System.out.println("weights3 : " + weights3);
            System.out.println("biases1 : " + biases1);
            System.out.println("biases2 : " + biases2);
            System.out.println("biases3 : " + biases3);
            System.out.println("velocity1 : " + velocity1);
            System.out.println("velocity2 : " + velocity2);
            System.out.println("velocity3 : " + velocity3);
            System.out.println("answer : " + (Math.pow(0.00000000000001, 100000)));

            if (Double.isNaN(categoricalCrossentropy(outputs, y_train))) {
                System.exit(0);
                main(args);
            }

            System.out.println("Past costs : " + pastCosts);*/
        }
        System.out.println("Past costs : " + pastCosts);
    }

    public static class DimensionalityReduction extends ScaledSigmoidMNIST {

        public static void reduceDimensions() {
            //first get variance of x_train
            ArrayList<Double> variance_data = new ArrayList<Double>();
            for (int column = 0; column < x_train.get(0).size(); column++) {
                ArrayList<Double> currentCol = new ArrayList<Double>();
                for (int i =0; i < x_train.size(); i++){currentCol.add(x_train.get(i).get(column));}
                variance_data.add(variance(currentCol));
            }

            //all columns to be removed in x_train
            ArrayList<Integer> removingColumns = new ArrayList<Integer>();
            for (int featureRemoved = 0; featureRemoved < 784 - numFeatures; featureRemoved++) {
                //go through the data find the minimum in entire set
                double minimum = Collections.min(variance_data);
                int removeCol = variance_data.indexOf(minimum);
                for (int input = 0; input  < x_train.size(); input++) {
                    x_train.get(input).remove(removeCol);
                }
            }

            variance_data = new ArrayList<Double>();
            for (int column = 0; column < x_train.get(0).size(); column++) {
                ArrayList<Double> currentCol = new ArrayList<Double>();
                for (int i =0; i < x_train.size(); i++){currentCol.add(x_train.get(i).get(column));}
                variance_data.add(variance(currentCol));
            }

            System.out.println("Variance of x_train : " + vectorSum(variance_data));
        }

        public static double vectorSum(ArrayList<Double> x) {
            double sum = 0;
            for (int i =0; i < x.size(); i++) {sum += x.get(i);}
            return sum;
        }


        public static double variance(ArrayList<Double> x) {
            //first get mean
            double meanSum = 0;
            for (int i = 0; i < x.size(); i++) {meanSum += x.get(i);}
            double mean = meanSum/x.size();

            double numerator = 0;
            for (int i = 0; i < x.size(); i++) {numerator += (x.get(i) - mean) * (x.get(i) - mean);}
            return numerator/x.size();
        }


        public static ArrayList<Double> vectorSubtract(ArrayList<Double>a1, ArrayList<Double>a2) {
            //here a1 + a2
            ArrayList<Double> answer = new ArrayList<Double>();
            for (int dim = 0 ; dim < a1.size(); dim ++) {
                answer.add(a1.get(dim) - a2.get(dim));
            }

            return answer;
        }

    }
}

