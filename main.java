import weka.core.Instance;
import weka.core.Instances;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.REPTree;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.Logistic;

public class main {

    private static final String COMMA_DELIMITER = ",";
    private static final String NEW_LINE_SEPARATOR = "\n";
    private static String FILE_HEADER = "DecisionTree, RandomForest, SupportVector, LogisticRegression, " +
            "K Nearest, Ensemble";

    /** Loads the specified data into an instances object and sets the class value
     * If the data has more than 2 classes, the most common class is set to 1 and all others are set to 0.
     *
     * @param filePath the path of the file containing the data to load
     * @return the data as an Instances object
     */
    public static Instances loadClassificationData(String filePath){
        Instances data = null;
        try{
            FileReader reader = new FileReader(filePath);
            data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);

            if (data.numClasses() > 2){
                int[] classes = new int[data.numClasses()];
                for (Instance i : data){
                    double classVal = i.classValue();
                    classes[(int)classVal] += 1;
                }

                int max = classes[0];
                int mostCommonClass = 0;

                for (int i = 0; i < classes.length; i++)
                {
                    if (max < classes[i])
                    {
                        max = classes[i];
                        mostCommonClass = i;
                    }
                }

                for (Instance i : data){
                    if (i.classValue() == (double)mostCommonClass){
                        i.setClassValue(1.0);
                    }
                    else {
                        i.setClassValue(0.0);
                    }
                }

            }
        }
        catch(Exception e){
            System.out.println("Exception caught: "+e);
        }

        return data;
    }

    /** Saves results of classifier comparison/evaluation tests to file
     *
     * @param filePath the destination file for writing
     * @param results the results to write to file
     */
    public static void saveToFile(String filePath, double[][] results){

        FileWriter fileWriter = null;

        try {
            fileWriter = new FileWriter(filePath);

            fileWriter.append(FILE_HEADER);

            fileWriter.append(NEW_LINE_SEPARATOR);

            for (int i = 0; i < results[0].length; i++){
                fileWriter.append(Double.toString(results[0][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[1][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[2][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[3][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[4][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[5][i]));
                fileWriter.append(NEW_LINE_SEPARATOR);
            }

        } catch (Exception e) {
            e.printStackTrace();

        } finally {
            try {
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /** Method to test and save various performance matrix on different classifiers to file
     * The classifiers being tested must be modified inside the method.
     *
     * @param trainingData the training data to use
     * @param testData the test data to use
     * @throws Exception catches unhandled Weka exceptions
     */
    public static void testAndSaveMetrics(String[] trainingData, String[] testData) throws Exception {
        double[][] accuracyResults = new double[6][testData.length];
        double[][] tprResults = new double[6][testData.length];
        double[][] tnrResults = new double[6][testData.length];
        double[][] balancedResults = new double[6][testData.length];
        double[][] timings = new double[6][testData.length];

        for (int i = 0; i < trainingData.length; i++){
            Instances train = loadClassificationData(trainingData[i]);
            Instances test = loadClassificationData(testData[i]);

            REPTree decisionTree = new REPTree();
            long start = System.nanoTime();
            decisionTree.buildClassifier(train);
            long stop = System.nanoTime();
            timings[0][i] = (stop - start)/1000000.0;

            RandomForest randomForest = new RandomForest();
            long start2 = System.nanoTime();
            randomForest.buildClassifier(train);
            long stop2 = System.nanoTime();
            timings[1][i] = (stop2 - start2)/1000000.0;

            Logistic logisticRegression = new Logistic();
            long start3 = System.nanoTime();
            logisticRegression.buildClassifier(train);
            long stop3 = System.nanoTime();
            timings[2][i] = (stop3 - start3)/1000000.0;

            SMO supportVector = new SMO();
            long start4 = System.nanoTime();
            supportVector.buildClassifier(train);
            long stop4 = System.nanoTime();
            timings[3][i] = (stop4 - start4)/1000000.0;

            IBk kNearest = new IBk();
            long start5 = System.nanoTime();
            kNearest.buildClassifier(train);
            long stop5 = System.nanoTime();
            timings[4][i] = (stop5 - start5)/1000000.0;

            LinearPerceptronEnsemble ensemble = new LinearPerceptronEnsemble();
            long start6 = System.nanoTime();
            ensemble.buildClassifier(train);
            long stop6 = System.nanoTime();
            timings[5][i] = (stop6 - start6)/1000000.0;

            double[] treePrediction = new double[test.numInstances()];
            double[] forestPrediction = new double[test.numInstances()];
            double[] supportPrediction = new double[test.numInstances()];
            double[] logisticPrediction = new double[test.numInstances()];
            double[] ibkPrediction = new double[test.numInstances()];
            double[] ensemblePrediction = new double[test.numInstances()];

            for (int j = 0; j < test.numInstances(); j++){
                treePrediction[j] = decisionTree.classifyInstance(test.instance(j));
                forestPrediction[j] = randomForest.classifyInstance(test.instance(j));
                supportPrediction[j] = supportVector.classifyInstance(test.instance(j));
                logisticPrediction[j] = logisticRegression.classifyInstance(test.instance(j));
                ibkPrediction[j] = kNearest.classifyInstance(test.instance(j));
                ensemblePrediction[j] = ensemble.classifyInstance(test.instance(j));
            }

            double treeAccuracy = EvaluationTools.calculateAccuracy(test, treePrediction);
            double forestAccuracy = EvaluationTools.calculateAccuracy(test, forestPrediction);
            double supportAccuracy = EvaluationTools.calculateAccuracy(test, supportPrediction);
            double logisticAccuracy = EvaluationTools.calculateAccuracy(test, logisticPrediction);
            double ibkAccuracy = EvaluationTools.calculateAccuracy(test, ibkPrediction);
            double ensembleAccuracy = EvaluationTools.calculateAccuracy(test, ensemblePrediction);

            accuracyResults[0][i] = treeAccuracy;
            accuracyResults[1][i] = forestAccuracy;
            accuracyResults[2][i] = supportAccuracy;
            accuracyResults[3][i] = logisticAccuracy;
            accuracyResults[4][i] = ibkAccuracy;
            accuracyResults[5][i] = ensembleAccuracy;

            double[][] treeCM = EvaluationTools.makeConfusionMatrix(test, treePrediction);
            double[][] forestCM = EvaluationTools.makeConfusionMatrix(test, forestPrediction);
            double[][] supportCM = EvaluationTools.makeConfusionMatrix(test, supportPrediction);
            double[][] logisticCM = EvaluationTools.makeConfusionMatrix(test, logisticPrediction);
            double[][] ibkCM = EvaluationTools.makeConfusionMatrix(test, ibkPrediction);
            double[][] ensembleCM = EvaluationTools.makeConfusionMatrix(test, ensemblePrediction);

            double treeTPR = EvaluationTools.calculateTPR(treeCM);
            double forestTPR = EvaluationTools.calculateTPR(forestCM);
            double supportTPR = EvaluationTools.calculateTPR(supportCM);
            double logisticTPR = EvaluationTools.calculateTPR(logisticCM);
            double ibkTPR = EvaluationTools.calculateTPR(ibkCM);
            double ensembleTPR = EvaluationTools.calculateTPR(ensembleCM);

            tprResults[0][i] = treeTPR;
            tprResults[1][i] = forestTPR;
            tprResults[2][i] = supportTPR;
            tprResults[3][i] = logisticTPR;
            tprResults[4][i] = ibkTPR;
            tprResults[5][i] = ensembleTPR;

            double treeTNR = EvaluationTools.calculateTNR(treeCM);
            double forestTNR = EvaluationTools.calculateTNR(forestCM);
            double supportTNR = EvaluationTools.calculateTNR(supportCM);
            double logisticTNR = EvaluationTools.calculateTNR(logisticCM);
            double ibkTNR = EvaluationTools.calculateTNR(ibkCM);
            double ensembleTNR = EvaluationTools.calculateTNR(ensembleCM);

            tnrResults[0][i] = treeTNR;
            tnrResults[1][i] = forestTNR;
            tnrResults[2][i] = supportTNR;
            tnrResults[3][i] = logisticTNR;
            tnrResults[4][i] = ibkTNR;
            tnrResults[5][i] = ensembleTNR;

            double treeBalanced = EvaluationTools.calculateBalancedAccuracy(treeTPR, treeTNR);
            double forestBalanced = EvaluationTools.calculateBalancedAccuracy(forestTPR, forestTNR);
            double supportBalanced = EvaluationTools.calculateBalancedAccuracy(supportTPR, supportTNR);
            double logisticBalanced = EvaluationTools.calculateBalancedAccuracy(logisticTPR, logisticTNR);
            double ibkBalanced = EvaluationTools.calculateBalancedAccuracy(ibkTPR, ibkTNR);
            double ensembleBalanced = EvaluationTools.calculateBalancedAccuracy(ensembleTPR, ensembleTNR);

            balancedResults[0][i] = treeBalanced;
            balancedResults[1][i] = forestBalanced;
            balancedResults[2][i] = supportBalanced;
            balancedResults[3][i] = logisticBalanced;
            balancedResults[4][i] = ibkBalanced;
            balancedResults[5][i] = ensembleBalanced;

        }

        saveToFile("accuracy4.csv", accuracyResults);
        saveToFile("tpr4.csv", tprResults);
        saveToFile("tnr4.csv", tnrResults);
        saveToFile("balanced4.csv", balancedResults);
        saveToFile("timings4.csv", timings);
    }

    /** Saves the case study results to file in a different format to other results files
     *
     * @param filePath the destination file for writing
     * @param results the results to write to file
     */
    public static void saveCaseStudy(String filePath, double[][] results){
        FileWriter fileWriter = null;

        try {
            fileWriter = new FileWriter(filePath);

            fileWriter.append(FILE_HEADER);

            fileWriter.append(NEW_LINE_SEPARATOR);
            for (int i = 0; i < results[0].length; i++){
                if (i == 0) {
                    fileWriter.append("Accuracy: ");
                }
                else if (i == 1){
                    fileWriter.append("Balanced Accuracy: ");
                }
                else if (i == 2){
                    fileWriter.append("TPR: ");
                }
                else if (i == 3){
                    fileWriter.append("TNR: ");
                }
                else if (i == 4){
                    fileWriter.append("Timings: ");
                }
                fileWriter.append(Double.toString(results[0][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[1][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[2][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[3][i]));
                fileWriter.append(COMMA_DELIMITER);
                fileWriter.append(Double.toString(results[4][i]));
                fileWriter.append(NEW_LINE_SEPARATOR);
            }

        } catch (Exception e) {
            e.printStackTrace();

        } finally {
            try {
                fileWriter.flush();
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /** Tests and saves various performance metrics for the case study on the twonorm dataset
     *
     * @param trainingData the training data to use
     * @param testData the test data to use
     * @throws Exception catches unhandled Weka exceptions
     */
    public static void testCaseStudy(String trainingData, String testData) throws Exception {
        double[] accuracyResults = new double[6];
        double[] tprResults = new double[6];
        double[] tnrResults = new double[6];
        double[] balancedResults = new double[6];
        double[] timings = new double[6];

        Instances train = loadClassificationData(trainingData);
        Instances test = loadClassificationData(testData);

        REPTree decisionTree = new REPTree();
        long start = System.nanoTime();
        decisionTree.buildClassifier(train);
        long stop = System.nanoTime();
        timings[0] = (stop - start) / 1000000.0;

        RandomForest randomForest = new RandomForest();
        long start2 = System.nanoTime();
        randomForest.buildClassifier(train);
        long stop2 = System.nanoTime();
        timings[1] = (stop2 - start2) / 1000000.0;

        Logistic logisticRegression = new Logistic();
        long start3 = System.nanoTime();
        logisticRegression.buildClassifier(train);
        long stop3 = System.nanoTime();
        timings[2] = (stop3 - start3) / 1000000.0;

        SMO supportVector = new SMO();
        long start4 = System.nanoTime();
        supportVector.buildClassifier(train);
        long stop4 = System.nanoTime();
        timings[3] = (stop4 - start4) / 1000000.0;

        IBk kNearest = new IBk();
        long start5 = System.nanoTime();
        kNearest.buildClassifier(train);
        long stop5 = System.nanoTime();
        timings[4] = (stop5 - start5) / 1000000.0;

        LinearPerceptronEnsemble ensemble = new LinearPerceptronEnsemble();
        long start6 = System.nanoTime();
        ensemble.buildClassifier(train);
        long stop6 = System.nanoTime();
        timings[5] = (stop6 - start6) / 1000000.0;

        double[] treePrediction = new double[test.numInstances()];
        double[] forestPrediction = new double[test.numInstances()];
        double[] supportPrediction = new double[test.numInstances()];
        double[] logisticPrediction = new double[test.numInstances()];
        double[] ibkPrediction = new double[test.numInstances()];
        double[] ensemblePrediction = new double[test.numInstances()];

        for (int j = 0; j < test.numInstances(); j++) {
            treePrediction[j] = decisionTree.classifyInstance(test.instance(j));
            forestPrediction[j] = randomForest.classifyInstance(test.instance(j));
            supportPrediction[j] = supportVector.classifyInstance(test.instance(j));
            logisticPrediction[j] = logisticRegression.classifyInstance(test.instance(j));
            ibkPrediction[j] = kNearest.classifyInstance(test.instance(j));
             ensemblePrediction[j] = ensemble.classifyInstance(test.instance(j));
        }

        double treeAccuracy = EvaluationTools.calculateAccuracy(test, treePrediction);
        double forestAccuracy = EvaluationTools.calculateAccuracy(test, forestPrediction);
        double supportAccuracy = EvaluationTools.calculateAccuracy(test, supportPrediction);
        double logisticAccuracy = EvaluationTools.calculateAccuracy(test, logisticPrediction);
        double ibkAccuracy = EvaluationTools.calculateAccuracy(test, ibkPrediction);
        double ensembleAccuracy = EvaluationTools.calculateAccuracy(test, ensemblePrediction);

        accuracyResults[0] = treeAccuracy;
        accuracyResults[1] = forestAccuracy;
        accuracyResults[2] = supportAccuracy;
        accuracyResults[3] = logisticAccuracy;
        accuracyResults[4] = ibkAccuracy;
        accuracyResults[5] = ensembleAccuracy;

        double[][] treeCM = EvaluationTools.makeConfusionMatrix(test, treePrediction);
        double[][] forestCM = EvaluationTools.makeConfusionMatrix(test, forestPrediction);
        double[][] supportCM = EvaluationTools.makeConfusionMatrix(test, supportPrediction);
        double[][] logisticCM = EvaluationTools.makeConfusionMatrix(test, logisticPrediction);
        double[][] ibkCM = EvaluationTools.makeConfusionMatrix(test, ibkPrediction);
        double[][] ensembleCM = EvaluationTools.makeConfusionMatrix(test, ensemblePrediction);

        double treeTPR = EvaluationTools.calculateTPR(treeCM);
        double forestTPR = EvaluationTools.calculateTPR(forestCM);
        double supportTPR = EvaluationTools.calculateTPR(supportCM);
        double logisticTPR = EvaluationTools.calculateTPR(logisticCM);
        double ibkTPR = EvaluationTools.calculateTPR(ibkCM);
        double ensembleTPR = EvaluationTools.calculateTPR(ensembleCM);

        tprResults[0] = treeTPR;
        tprResults[1] = forestTPR;
        tprResults[2] = supportTPR;
        tprResults[3] = logisticTPR;
        tprResults[4] = ibkTPR;
        tprResults[5] = ensembleTPR;

        double treeTNR = EvaluationTools.calculateTNR(treeCM);
        double forestTNR = EvaluationTools.calculateTNR(forestCM);
        double supportTNR = EvaluationTools.calculateTNR(supportCM);
        double logisticTNR = EvaluationTools.calculateTNR(logisticCM);
        double ibkTNR = EvaluationTools.calculateTNR(ibkCM);
        double ensembleTNR = EvaluationTools.calculateTNR(ensembleCM);

        tnrResults[0] = treeTNR;
        tnrResults[1] = forestTNR;
        tnrResults[2] = supportTNR;
        tnrResults[3] = logisticTNR;
        tnrResults[4] = ibkTNR;
        tnrResults[5] = ensembleTNR;

        double treeBalanced = EvaluationTools.calculateBalancedAccuracy(treeTPR, treeTNR);
        double forestBalanced = EvaluationTools.calculateBalancedAccuracy(forestTPR, forestTNR);
        double supportBalanced = EvaluationTools.calculateBalancedAccuracy(supportTPR, supportTNR);
        double logisticBalanced = EvaluationTools.calculateBalancedAccuracy(logisticTPR, logisticTNR);
        double ibkBalanced = EvaluationTools.calculateBalancedAccuracy(ibkTPR, ibkTNR);
        double ensembleBalanced = EvaluationTools.calculateBalancedAccuracy(ensembleTPR, ensembleTNR);

        balancedResults[0] = treeBalanced;
        balancedResults[1] = forestBalanced;
        balancedResults[2] = supportBalanced;
        balancedResults[3] = logisticBalanced;
        balancedResults[4] = ibkBalanced;
        balancedResults[5] = ensembleBalanced;

        double[][] results = {accuracyResults, balancedResults, tprResults, tnrResults, timings};
        saveCaseStudy("caseStudy.csv", results);

    }

    /** Helper method to print the number of instances in the datasets to console
     *
     * @param data array of filepaths to data sources
     */
    public static void numInstances(String[] data){
        for (String s : data) {
            Instances i = loadClassificationData(s);
            System.out.println(i.numInstances());
        }
    }

    /** Helper method to print class distributions for all datasets to console
     *
     * @param data array of filepaths to data sources
     */
    public static void classDistrib(String[] data){

        DecimalFormat dec = new DecimalFormat("#.###");

        for (String s : data){
            Instances in = loadClassificationData(s);
            int[] classes = new int[in.numClasses()];
            for (Instance instance : in){
                classes[(int)instance.classValue()] += 1;
            }
            double[] classDistrib = new double[in.numClasses()];
            for (int i = 0; i < classDistrib.length; i++){
                classDistrib[i] = (double) classes[i] / in.numInstances();
            }

            for (double d : classDistrib){
                System.out.print(dec.format(d) + ", ");
            }

            System.out.println();
        }
    }

    public static void main(String[] args) throws Exception {

        //Testing carried out
        /*String[] trainingData = {"UCIContinuous/bank/bank_TRAIN.arff", "UCIContinuous/blood/blood_TRAIN.arff",
                "UCIContinuous/breast-cancer-wisc-diag/breast-cancer-wisc-diag_TRAIN.arff",
                "UCIContinuous/breast-tissue/breast-tissue_TRAIN.arff",
                "UCIContinuous/cardiotocography-10clases/cardiotocography-10clases_TRAIN.arff",
                "UCIContinuous/conn-bench-sonar-mines-rocks/conn-bench-sonar-mines-rocks_TRAIN.arff",
                "UCIContinuous/conn-bench-vowel-deterding/conn-bench-vowel-deterding_TRAIN.arff",
                "UCIContinuous/ecoli/ecoli_TRAIN.arff",
                "UCIContinuous/glass/glass_TRAIN.arff",
                "UCIContinuous/hill-valley/hill-valley_TRAIN.arff",
                "UCIContinuous/image-segmentation/image-segmentation_TRAIN.arff",
                "UCIContinuous/ionosphere/ionosphere_TRAIN.arff",
                "UCIContinuous/iris/iris_TRAIN.arff",
                "UCIContinuous/libras/libras_TRAIN.arff",
                "UCIContinuous/oocytes_merluccius_nucleus_4d/oocytes_merluccius_nucleus_4d_TRAIN.arff",
                "UCIContinuous/oocytes_trisopterus_states_5b/oocytes_trisopterus_states_5b_TRAIN.arff",
                "UCIContinuous/optical/optical_TRAIN.arff",
                "UCIContinuous/ozone/ozone_TRAIN.arff",
                "UCIContinuous/page-blocks/page-blocks_TRAIN.arff",
                "UCIContinuous/parkinsons/parkinsons_TRAIN.arff",
                "UCIContinuous/planning/planning_TRAIN.arff",
                "UCIContinuous/post-operative/post-operative_TRAIN.arff",
                "UCIContinuous/ringnorm/ringnorm_TRAIN.arff",
                "UCIContinuous/seeds/seeds_TRAIN.arff",
                "UCIContinuous/spambase/spambase_TRAIN.arff",
                "UCIContinuous/statlog-landsat/statlog-landsat_TRAIN.arff",
                "UCIContinuous/statlog-vehicle/statlog-vehicle_TRAIN.arff",
                "UCIContinuous/steel-plates/steel-plates_TRAIN.arff",
                "UCIContinuous/synthetic-control/synthetic-control_TRAIN.arff",
                "UCIContinuous/twonorm/twonorm_TRAIN.arff",
                "UCIContinuous/vertebral-column-3clases/vertebral-column-3clases_TRAIN.arff",
                "UCIContinuous/wall-following/wall-following_TRAIN.arff",
                "UCIContinuous/waveform-noise/waveform-noise_TRAIN.arff",
                "UCIContinuous/wine-quality-white/wine-quality-white_TRAIN.arff",
                "UCIContinuous/yeast/yeast_TRAIN.arff"
        };

        String[] testData = {"UCIContinuous/bank/bank_TEST.arff", "UCIContinuous/blood/blood_TEST.arff",
                "UCIContinuous/breast-cancer-wisc-diag/breast-cancer-wisc-diag_TEST.arff",
                "UCIContinuous/breast-tissue/breast-tissue_TEST.arff",
                "UCIContinuous/cardiotocography-10clases/cardiotocography-10clases_TEST.arff",
                "UCIContinuous/conn-bench-sonar-mines-rocks/conn-bench-sonar-mines-rocks_TEST.arff",
                "UCIContinuous/conn-bench-vowel-deterding/conn-bench-vowel-deterding_TEST.arff",
                "UCIContinuous/ecoli/ecoli_TEST.arff",
                "UCIContinuous/glass/glass_TEST.arff",
                "UCIContinuous/hill-valley/hill-valley_TEST.arff",
                "UCIContinuous/image-segmentation/image-segmentation_TEST.arff",
                "UCIContinuous/ionosphere/ionosphere_TEST.arff",
                "UCIContinuous/iris/iris_TEST.arff",
                "UCIContinuous/libras/libras_TEST.arff",
                "UCIContinuous/oocytes_merluccius_nucleus_4d/oocytes_merluccius_nucleus_4d_TEST.arff",
                "UCIContinuous/oocytes_trisopterus_states_5b/oocytes_trisopterus_states_5b_TEST.arff",
                "UCIContinuous/optical/optical_TEST.arff",
                "UCIContinuous/ozone/ozone_TEST.arff",
                "UCIContinuous/page-blocks/page-blocks_TEST.arff",
                "UCIContinuous/parkinsons/parkinsons_TEST.arff",
                "UCIContinuous/planning/planning_TEST.arff",
                "UCIContinuous/post-operative/post-operative_TEST.arff",
                "UCIContinuous/ringnorm/ringnorm_TEST.arff",
                "UCIContinuous/seeds/seeds_TEST.arff",
                "UCIContinuous/spambase/spambase_TEST.arff",
                "UCIContinuous/statlog-landsat/statlog-landsat_TEST.arff",
                "UCIContinuous/statlog-vehicle/statlog-vehicle_TEST.arff",
                "UCIContinuous/steel-plates/steel-plates_TEST.arff",
                "UCIContinuous/synthetic-control/synthetic-control_TEST.arff",
                "UCIContinuous/twonorm/twonorm_TEST.arff",
                "UCIContinuous/vertebral-column-3clases/vertebral-column-3clases_TEST.arff",
                "UCIContinuous/wall-following/wall-following_TEST.arff",
                "UCIContinuous/waveform-noise/waveform-noise_TEST.arff",
                "UCIContinuous/wine-quality-white/wine-quality-white_TEST.arff",
                "UCIContinuous/yeast/yeast_TEST.arff"};*/

        //testAndSaveMetrics(trainingData, testData);

        //String[] trainingData = "UCIContinuous/twonorm/twonorm_TRAIN.arff";
        //String[] testData = "UCIContinuous/twonorm/twonorm_TEST.arff";

        //testAndSaveMetrics(trainingData, testData);
    }
}
