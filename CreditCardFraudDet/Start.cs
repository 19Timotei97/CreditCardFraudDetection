using System;
using System.IO;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Drawing;

namespace CreditCardFraudDet
{
    public partial class Start : Form
    {
        string pathToDataSet = string.Empty;

        private readonly string DecTree = "DECISION_TREE";
        private readonly string RandForest = "RANDOM_FOREST";
        private readonly string NaiveBayes = "NAIVE_BAYES";
        private readonly string NeuralNetwork = "NEURAL_NETWORK";
        private readonly string allAlgs = "ALL";
        private readonly Dictionary<RadioButton, string> MapButtons = new Dictionary<RadioButton, string>();
        private readonly Dictionary<RadioButton, string> MapDatasetButtons = new Dictionary<RadioButton, string>();

        string pythonScript;
        private RadioButton selectedAlgorithm;
        private RadioButton selectedDataset;

        int minimumTestSize = 5;
        int maximumTestSize = 70;
        private string datasetTrainSize = "15";

        string pythonScriptForPCA = @"\CreditCardFraudDetection_PCA.py";
        readonly string pythonScriptForGenerated = @"\CreditCardFraudDetection_generated.py";

        public string DatasetTrainSize { get => datasetTrainSize; set => datasetTrainSize = value; }

        public Start() => InitializeComponent();

        private void FraudDet_Click(object sender, EventArgs e) => Run_cmd();

        private void RunAlgorithms(String Algorithm)
        {
            SetPythonScript(selectedDataset);

            Directory.SetCurrentDirectory(Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..")));

            try
            {
                Process p = new Process
                {
                    // Change this according to your python.exe path.

                    // If you don't want a static location of the python executable
                    // you need to read the correct registry for it. The problem is that you most probably
                    // need Admin rights, on the PC that this app is run :(
                    StartInfo = new ProcessStartInfo(
                        @"F:\Progamming\Python37\python.exe", "\"" + Directory.GetCurrentDirectory() + pythonScript + "\"" + " " + pathToDataSet + " " + Algorithm + " " + DatasetTrainSize)
                    {
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardError = true // Any error in standard output will be redirected back (exceptions)
                    }
                };

                p.Start(); // For some reason, it doesnt work with graphviz, investigate!
                Cursor = Cursors.WaitCursor;

                string pythonScriptOutput = p.StandardOutput.ReadToEnd(); // Store the output of the Python script
                string errors = p.StandardError.ReadToEnd(); // Store the error for debugging

                debug.Text = errors;

                p.WaitForExit();

                AlgResult.Clear();

                AlgResult.Text = pythonScriptOutput;

                AlgResult.ReadOnly = true;

                AlgResult.ScrollBars = ScrollBars.Vertical;

                debug.Clear();

                debug.ScrollBars = ScrollBars.Vertical;

                Cursor = Cursors.Arrow;

                if (DecTree.Equals(Algorithm))
                {
                    MapDatasetButtons.TryGetValue(selectedDataset, out string type);

                    if (true == type.Contains("PCA"))
                    {
                        Image img = Image.FromFile(Directory.GetCurrentDirectory() + @"\decisionTree_fraudDet_PCA.png");

                        DecisionTreeBuilt imgShow = new DecisionTreeBuilt();

                        imgShow.ShowImage(img);

                        imgShow.Show();


                        Image prunedImgF1 = Image.FromFile(Directory.GetCurrentDirectory() + @"\decisionTree_fraudDet_PCA_pruned_f1.png");

                        DecisionTreePrunedF1 prunedImgF1Show = new DecisionTreePrunedF1();

                        prunedImgF1Show.ShowImage(prunedImgF1);

                        prunedImgF1Show.Show();


                        Image prunedImgAROC = Image.FromFile(Directory.GetCurrentDirectory() + @"\decisionTree_fraudDet_PCA_pruned_roc_auc.png");

                        DecisionTreePrunedAROC prunedImgAROCShow = new DecisionTreePrunedAROC();

                        prunedImgAROCShow.ShowImage(prunedImgF1);

                        prunedImgAROCShow.Show();
                    }

                    if (true == type.Contains("generated"))
                    {
                        Image img = Image.FromFile(Directory.GetCurrentDirectory() + @"\decisionTree_fraudDet_generated.png");

                        DecisionTreeBuilt imgShow = new DecisionTreeBuilt();

                        imgShow.ShowImage(img);

                        imgShow.Show();


                        Image prunedImgF1 = Image.FromFile(Directory.GetCurrentDirectory() + @"\decisionTree_fraudDet_generated_pruned_f1.png");

                        DecisionTreePrunedF1 prunedImgF1Show = new DecisionTreePrunedF1();

                        prunedImgF1Show.ShowImage(prunedImgF1);

                        prunedImgF1Show.Show();


                        Image prunedImgAROC = Image.FromFile(Directory.GetCurrentDirectory() + @"\decisionTree_fraudDet_generated_pruned_roc_auc.png");

                        DecisionTreePrunedAROC prunedImgAROCShow = new DecisionTreePrunedAROC();

                        prunedImgAROCShow.ShowImage(prunedImgF1);

                        prunedImgAROCShow.Show();
                    }
                }

                if (RandForest.Equals(Algorithm))
                {
                    MapDatasetButtons.TryGetValue(selectedDataset, out string type);

                    if (true == type.Contains("PCA"))
                    {
                        Image img = Image.FromFile(Directory.GetCurrentDirectory() + @"\randomForest_middle_fraudDet_PCA.png");

                        RandomForestMiddleTree imgShow = new RandomForestMiddleTree();

                        imgShow.ShowImage(img);

                        imgShow.Show();


                        Image prunedImgF1 = Image.FromFile(Directory.GetCurrentDirectory() + @"\randomForest_middle_fraudDet_pca_f1.png");

                        RandomForestMiddleF1 prunedImgF1Show = new RandomForestMiddleF1();

                        prunedImgF1Show.ShowImage(prunedImgF1);

                        prunedImgF1Show.Show();


                        Image prunedImgAROC = Image.FromFile(Directory.GetCurrentDirectory() + @"\randomForest_middle_fraudDet_pca_roc_auc.png");

                        RandomForestMiddleAROC prunedImgAROCShow = new RandomForestMiddleAROC();

                        prunedImgAROCShow.ShowImage(prunedImgF1);

                        prunedImgAROCShow.Show();
                    }

                    if (true == type.Contains("generated"))
                    {
                        Image img = Image.FromFile(Directory.GetCurrentDirectory() + @"\randomForest_middle_fraudDet_generated.png");

                        RandomForestMiddleTree imgShow = new RandomForestMiddleTree();

                        imgShow.ShowImage(img);

                        imgShow.Show();


                        Image prunedImgF1 = Image.FromFile(Directory.GetCurrentDirectory() + @"\randomForest_middle_fraudDet_generated_f1.png");

                        RandomForestMiddleF1 prunedImgF1Show = new RandomForestMiddleF1();

                        prunedImgF1Show.ShowImage(prunedImgF1);

                        prunedImgF1Show.Show();


                        Image prunedImgAROC = Image.FromFile(Directory.GetCurrentDirectory() + @"\randomForest_middle_fraudDet_generated_roc_auc.png");

                        RandomForestMiddleAROC prunedImgAROCShow = new RandomForestMiddleAROC();

                        prunedImgAROCShow.ShowImage(prunedImgF1);

                        prunedImgAROCShow.Show();
                    }
                }

                MessageBox.Show("The algorithm(s) finished execution.\nResults available in the corresponding TextBox.");

            }
            catch (Exception ex)
            {
                string exceptionMessage = ex.Message;
                debug.Text = "Exception thrown!\n " + exceptionMessage;
            }
        }

        private void Aglorithm_CheckedChanged(object sender, EventArgs evAr)
        {
            if (sender is RadioButton radBut)
            {
                // Ensure that the button was checked
                if (radBut.Checked)
                {
                    // Memorize which button was checked by saving a reference
                    selectedAlgorithm = radBut;
                }
            }
            else
            {
                MessageBox.Show("Something went horribly wrong with the aglorithm selection :( Please try again!");
                return;
            }
        }

        void DatasetType_CheckedChanged(object sender, EventArgs evAr)
        {
            if (!(sender is RadioButton radBut))
            {
                MessageBox.Show("Something went horribly wrong with the dataset type selection :( Please try again!");
                return;
            }
            else
            {
                // Make sure that the button was checked
                if (radBut.Checked)
                {
                    // Save a reference to that button
                    selectedDataset = radBut;
                }
            }
        }

        private void RunAlgSelected(RadioButton rb)
        {

            if (!MapButtons.TryGetValue(rb, out string selAlg))
            {
                MessageBox.Show("Please select the desired algorithm! If not sure, choose \"Compare all\"");
            }
            else
            {
                if (allAlgs.Equals(selAlg))
                {
                    MessageBox.Show("You have chosen to apply all of the clasification algorithms. Note that the execution time will be significant, please wait for the process to finish!");
                }
                _ = MessageBox.Show("Chosen algorithm(s): " + selAlg + ", train size: " + DatasetTrainSize + "." + "\nAlgorithm is starting classifying the dataset...");
                RunAlgorithms(selAlg);
            }
        }

        private void SetPythonScript(RadioButton rb)
        {
            if (!MapDatasetButtons.TryGetValue(rb, out pythonScript))
            {
                MessageBox.Show("Please select the dataset type in order to start the classification.");
            }
        }

        private void Run_cmd()
        {
            // Add the aglorithm types to the dictionary
            if (!MapButtons.ContainsKey(decTree))
            {
                MapButtons.Add(decTree, DecTree);
            }

            if (!MapButtons.ContainsKey(randForest))
            {
                MapButtons.Add(randForest, RandForest);
            }

            if (!MapButtons.ContainsKey(nBayes))
            {
                MapButtons.Add(nBayes, NaiveBayes);
            }

            if (!MapButtons.ContainsKey(nNet))
            {
                MapButtons.Add(nNet, NeuralNetwork);
            }

            if (!MapButtons.ContainsKey(compAll))
            {
                MapButtons.Add(compAll, allAlgs);
            }

            // Add the dataset types to the dictionary
            if (!MapDatasetButtons.ContainsKey(key: datasetWithPCA))
            {
                MapDatasetButtons.Add(key: datasetWithPCA, pythonScriptForPCA);
            }

            if (!MapDatasetButtons.ContainsKey(datasetGenerated))
            {
                MapDatasetButtons.Add(datasetGenerated, pythonScriptForGenerated);
            }

            if (String.IsNullOrEmpty(pathToDataSet))
            {
                MessageBox.Show("Dataset not inserted yet! Add a proper dataset and try again!");
            }
            else if (String.IsNullOrEmpty(trainDatasetSize.Text))
            {
                MessageBox.Show("Test size not inserted!");

                var selectedOption = MessageBox.Show("Would you like to use the default test size?", "Default test size", MessageBoxButtons.YesNo, MessageBoxIcon.Question);

                if (DialogResult.Yes == selectedOption)
                {
                    // RunAlgSelected(selectedAlgorithm);
                    trainDatasetSize.Text = datasetTrainSize;
                }
                else
                {
                    MessageBox.Show("Please add a size for the test dataset!");
                }
            }
            else
            {
                Regex testSize = new Regex("^[0-9][0-9]?$");

                if (testSize.IsMatch(trainDatasetSize.Text))
                {
                    int datasetDimension = Int16.Parse(trainDatasetSize.Text);

                    if (datasetDimension < minimumTestSize || datasetDimension > maximumTestSize)
                    {
                        MessageBox.Show("Please enter a value between the minimum and the maximum values.");
                    }
                    else
                    {
                        DatasetTrainSize = (Convert.ToDouble(datasetDimension) / 100).ToString();

                        RunAlgSelected(selectedAlgorithm);
                    }
                }
                else
                {
                    MessageBox.Show("Please enter digits only for the train size!");
                }
            }
        }
        private void ClearData_Click(object sender, EventArgs e)
        {
            pathToDataSet = string.Empty;
            testDataSetName.Clear();
        }

        private void UploadData_Click(object sender, EventArgs e)
        {
            OpenFileDialog oFd = new OpenFileDialog
            {
                Filter = "CSV files | *.csv", // Accepted file type

                Multiselect = false // Deny multiple upload, one data set at a time
            };

            if (DialogResult.OK == oFd.ShowDialog())
            {
                // Set the path of the file
                pathToDataSet = oFd.FileName;

                // Only put the name of the file for the user to know, not the whole path
                testDataSetName.Text = pathToDataSet.Substring(pathToDataSet.LastIndexOf("\\") + 1, (pathToDataSet.Length - pathToDataSet.LastIndexOf("\\")) - 1);
                testDataSetName.ReadOnly = true;
            }
        }

        private void CloseApp_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void ClearResults_Click(object sender, EventArgs e)
        {
            AlgResult.Clear();
        }

        private void SelectedAlgBut_Click(object sender, EventArgs e)
        {
            try
            {
                MessageBox.Show("Selected algorithm is: " + selectedAlgorithm.Text);
            }
            catch
            {
                MessageBox.Show("Didn't select any algorithm!");
            }
            
        }

        private void SelectedDatasetType_Click(object sender, EventArgs e)
        {
            try
            {
                MessageBox.Show("Selected dataset type is: " + selectedDataset.Text);
            }
            catch
            {
                MessageBox.Show("Didn't select any dataset type!");
            }
            
        }
    }
}
