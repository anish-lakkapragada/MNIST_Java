import java.io.BufferedReader;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.util.*;
public class NormalizationMNIST {
    private static double alpha = 1;
    public static ArrayList<ArrayList<Double>> x_train = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> y_train = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> x_test = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> y_test = new ArrayList<ArrayList<Double>>();
    public static int numData = 20000;
    public static int numVal = 75;
    public static int numNeurons1 = 32;
    public static int numNeurons2 = 16;
    public static int numNeurons3 = 10;
    public static double learningRate = 0.0000005;
    public static double biasLearningRate = 0.002;
    public static double momentum = 0;
    public static double biasMomentum = 0;
    public static double gradClip = 10000;
    public static int numEpochs = 100000;
    public static boolean useNesterov = true;
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
    public static ArrayList<ArrayList<Double>> pastCorrects = new ArrayList<ArrayList<Double>>();
    public static double mean = 0.0;
    public static void getTrainData() throws Exception {
        String train_path = "/Users/anish/Java Fun/ML Java/src/mnist_train.csv";
        String line = "";
        BufferedReader br = new BufferedReader(new FileReader(train_path));
        int z = numData;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            ArrayList<Double> currentX_train = new ArrayList<Double>();
            for (int i = 0; i < values.length; i++) {
                if (i == 0) {y_train.add(oneHot(Double.parseDouble(values[0])));}
                else {
                    currentX_train.add((Double.parseDouble(values[i])));
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
                if (i == 0) {y_test.add(oneHot(Double.parseDouble(values[0])));}
                else {
                    currentX_train.add((Double.parseDouble(values[i])));
                }
            }
            x_test.add(currentX_train);
            z -= 1;
            if (z == 0) {break;}
        }
    }

    public static void main(String[] args) throws Exception {
        getTrainData();
        getValidationData();
        normalizedData(x_train);
    }


    public static void normalizedData(ArrayList<ArrayList<Double>> x_train) {
        ArrayList<Double> x_trainVM = vectorMean(x_train);
        ArrayList<Double> variance = new ArrayList<Double>();
        for (int i = 0; i < x_train.get(0).size(); i++) {variance.add(0.0);}
        for (int i =0; i < x_train.size(); i++) {
            variance = vectorAdd(variance, scalarDivide(vectorSquare(vectorSubtract(x_train.get(i), x_trainVM)), x_train.size()));
        }

        double e = 0.00001;
        for (int i = 0; i < x_train.size(); i++) {
            x_train.set(i, elementWiseDivision(vectorSubtract(x_train.get(i), x_trainVM), vectorSqrt(scalarAdd(variance, e))));
        }

        System.out.println("x_train : " + x_train);

    }


    public static ArrayList<Double> oneHot(double x) {
        ArrayList<Double> oneHotted = new ArrayList<Double>();
        for (int deep = 0; deep < 10; deep++) {
            if (deep == x) {oneHotted.add(1.0);}
            else {oneHotted.add(0.0);}
        }
        return oneHotted;
    }

    public static ArrayList<Double> vectorMean(ArrayList<ArrayList<Double>> x) {
        ArrayList<Double> vm = new ArrayList<Double>();
        for (int i =0; i < x.get(0).size(); i++) {
            vm.add(0.0);
        }
        for (int i =0; i < x.size(); i++) {
            vm = vectorAdd(vm, x.get(i));
        }
        return scalarDivide(vm, (double) x.size());
    }

    public static ArrayList<Double> vectorSquare(ArrayList<Double> x) {
        ArrayList<Double> z= new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i) * x.get(i));
        }
        return z;
    }

    public static ArrayList<Double> vectorSqrt(ArrayList<Double> x) {
        ArrayList<Double> z= new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(Math.sqrt(x.get(i)));
        }
        return z;
    }

    public static ArrayList<Double> vectorSubtract(ArrayList<Double> x, ArrayList<Double> y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i) - y.get(i));
        }
        return z;
    }

    public static ArrayList<Double> scalarDivide(ArrayList<Double> x, double y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i)/y);
        }
        return z;
    }

    public static ArrayList<Double> scalarAdd(ArrayList<Double> x, double y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i) + y);
        }
        return z;
    }

    public static ArrayList<Double> elementWiseDivision(ArrayList<Double> x, ArrayList<Double> y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i)/y.get(i));
        }
        return z;
    }


    public static ArrayList<Double> vectorAdd(ArrayList<Double> x, ArrayList<Double> y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i) + y.get(i));
        }
        return z;
    }



}