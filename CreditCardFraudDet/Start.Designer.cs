namespace CreditCardFraudDet
{
    partial class Start
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.label1 = new System.Windows.Forms.Label();
            this.appName = new System.Windows.Forms.Label();
            this.FraudDet = new System.Windows.Forms.Button();
            this.ClearData = new System.Windows.Forms.Button();
            this.AddData = new System.Windows.Forms.Button();
            this.CloseApp = new System.Windows.Forms.Button();
            this.AlgResult = new System.Windows.Forms.TextBox();
            this.algRes = new System.Windows.Forms.Label();
            this.testDataSetName = new System.Windows.Forms.TextBox();
            this.datasetName = new System.Windows.Forms.Label();
            this.ClearResults = new System.Windows.Forms.Button();
            this.selAlg = new System.Windows.Forms.Label();
            this.decTree = new System.Windows.Forms.RadioButton();
            this.randForest = new System.Windows.Forms.RadioButton();
            this.nBayes = new System.Windows.Forms.RadioButton();
            this.nNet = new System.Windows.Forms.RadioButton();
            this.compAll = new System.Windows.Forms.RadioButton();
            this.testPercentage = new System.Windows.Forms.Label();
            this.trainDatasetSize = new System.Windows.Forms.TextBox();
            this.minValue = new System.Windows.Forms.Label();
            this.maxValue = new System.Windows.Forms.Label();
            this.percentOftestSize = new System.Windows.Forms.Label();
            this.datasetWithPCA = new System.Windows.Forms.RadioButton();
            this.datasetGenerated = new System.Windows.Forms.RadioButton();
            this.datasetType = new System.Windows.Forms.Label();
            this.groupBoxAlgorithms = new System.Windows.Forms.GroupBox();
            this.selectedAlgBut = new System.Windows.Forms.Button();
            this.groupBoxDatasetType = new System.Windows.Forms.GroupBox();
            this.selectedDatasetType = new System.Windows.Forms.Button();
            this.debug = new System.Windows.Forms.TextBox();
            this.groupBoxAlgorithms.SuspendLayout();
            this.groupBoxDatasetType.SuspendLayout();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(0, 0);
            this.label1.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(75, 19);
            this.label1.TabIndex = 0;
            // 
            // appName
            // 
            this.appName.AutoSize = true;
            this.appName.Font = new System.Drawing.Font("Microsoft Sans Serif", 13.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.appName.Location = new System.Drawing.Point(9, 19);
            this.appName.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.appName.Name = "appName";
            this.appName.Size = new System.Drawing.Size(234, 24);
            this.appName.TabIndex = 1;
            this.appName.Text = "Credit Card Fraud Detector";
            // 
            // FraudDet
            // 
            this.FraudDet.Location = new System.Drawing.Point(9, 602);
            this.FraudDet.Margin = new System.Windows.Forms.Padding(2);
            this.FraudDet.Name = "FraudDet";
            this.FraudDet.Size = new System.Drawing.Size(220, 57);
            this.FraudDet.TabIndex = 3;
            this.FraudDet.Text = "Find fraudulent transactions";
            this.FraudDet.UseVisualStyleBackColor = true;
            this.FraudDet.Click += new System.EventHandler(this.FraudDet_Click);
            // 
            // ClearData
            // 
            this.ClearData.Location = new System.Drawing.Point(226, 57);
            this.ClearData.Margin = new System.Windows.Forms.Padding(2);
            this.ClearData.Name = "ClearData";
            this.ClearData.Size = new System.Drawing.Size(146, 37);
            this.ClearData.TabIndex = 4;
            this.ClearData.Text = "Clear inserted dataset";
            this.ClearData.UseVisualStyleBackColor = true;
            this.ClearData.Click += new System.EventHandler(this.ClearData_Click);
            // 
            // AddData
            // 
            this.AddData.Location = new System.Drawing.Point(9, 57);
            this.AddData.Margin = new System.Windows.Forms.Padding(2);
            this.AddData.Name = "AddData";
            this.AddData.Size = new System.Drawing.Size(161, 37);
            this.AddData.TabIndex = 5;
            this.AddData.Text = "Add dataset";
            this.AddData.UseVisualStyleBackColor = true;
            this.AddData.Click += new System.EventHandler(this.UploadData_Click);
            // 
            // CloseApp
            // 
            this.CloseApp.Location = new System.Drawing.Point(907, 602);
            this.CloseApp.Margin = new System.Windows.Forms.Padding(2);
            this.CloseApp.Name = "CloseApp";
            this.CloseApp.Size = new System.Drawing.Size(156, 57);
            this.CloseApp.TabIndex = 6;
            this.CloseApp.Text = "Close";
            this.CloseApp.UseVisualStyleBackColor = true;
            this.CloseApp.Click += new System.EventHandler(this.CloseApp_Click);
            // 
            // AlgResult
            // 
            this.AlgResult.Location = new System.Drawing.Point(394, 67);
            this.AlgResult.Margin = new System.Windows.Forms.Padding(2);
            this.AlgResult.Multiline = true;
            this.AlgResult.Name = "AlgResult";
            this.AlgResult.Size = new System.Drawing.Size(669, 384);
            this.AlgResult.TabIndex = 7;
            // 
            // algRes
            // 
            this.algRes.AutoSize = true;
            this.algRes.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.algRes.Location = new System.Drawing.Point(391, 36);
            this.algRes.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.algRes.Name = "algRes";
            this.algRes.Size = new System.Drawing.Size(141, 20);
            this.algRes.TabIndex = 8;
            this.algRes.Text = "Algorithm(s) result:";
            // 
            // testDataSetName
            // 
            this.testDataSetName.Location = new System.Drawing.Point(155, 135);
            this.testDataSetName.Margin = new System.Windows.Forms.Padding(2);
            this.testDataSetName.Name = "testDataSetName";
            this.testDataSetName.Size = new System.Drawing.Size(218, 20);
            this.testDataSetName.TabIndex = 9;
            // 
            // datasetName
            // 
            this.datasetName.AutoSize = true;
            this.datasetName.Font = new System.Drawing.Font("Microsoft Sans Serif", 10.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.datasetName.Location = new System.Drawing.Point(6, 133);
            this.datasetName.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.datasetName.Name = "datasetName";
            this.datasetName.Size = new System.Drawing.Size(122, 18);
            this.datasetName.TabIndex = 10;
            this.datasetName.Text = "Dataset file name";
            // 
            // ClearResults
            // 
            this.ClearResults.Location = new System.Drawing.Point(394, 602);
            this.ClearResults.Margin = new System.Windows.Forms.Padding(2);
            this.ClearResults.Name = "ClearResults";
            this.ClearResults.Size = new System.Drawing.Size(160, 57);
            this.ClearResults.TabIndex = 14;
            this.ClearResults.Text = "ClearResults";
            this.ClearResults.UseVisualStyleBackColor = true;
            this.ClearResults.Click += new System.EventHandler(this.ClearResults_Click);
            // 
            // selAlg
            // 
            this.selAlg.AutoSize = true;
            this.selAlg.Font = new System.Drawing.Font("Microsoft Sans Serif", 10.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.selAlg.Location = new System.Drawing.Point(6, 191);
            this.selAlg.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.selAlg.Name = "selAlg";
            this.selAlg.Size = new System.Drawing.Size(352, 18);
            this.selAlg.TabIndex = 17;
            this.selAlg.Text = "Select your desired algorithm or compare all of them";
            // 
            // decTree
            // 
            this.decTree.AutoSize = true;
            this.decTree.Location = new System.Drawing.Point(5, 17);
            this.decTree.Margin = new System.Windows.Forms.Padding(2);
            this.decTree.Name = "decTree";
            this.decTree.Size = new System.Drawing.Size(91, 17);
            this.decTree.TabIndex = 18;
            this.decTree.TabStop = true;
            this.decTree.Text = "Decision Tree";
            this.decTree.UseVisualStyleBackColor = true;
            this.decTree.CheckedChanged += new System.EventHandler(this.Aglorithm_CheckedChanged);
            // 
            // randForest
            // 
            this.randForest.AutoSize = true;
            this.randForest.Location = new System.Drawing.Point(118, 17);
            this.randForest.Margin = new System.Windows.Forms.Padding(2);
            this.randForest.Name = "randForest";
            this.randForest.Size = new System.Drawing.Size(97, 17);
            this.randForest.TabIndex = 19;
            this.randForest.TabStop = true;
            this.randForest.Text = "Random Forest";
            this.randForest.UseVisualStyleBackColor = true;
            this.randForest.CheckedChanged += new System.EventHandler(this.Aglorithm_CheckedChanged);
            // 
            // nBayes
            // 
            this.nBayes.AutoSize = true;
            this.nBayes.Location = new System.Drawing.Point(250, 17);
            this.nBayes.Margin = new System.Windows.Forms.Padding(2);
            this.nBayes.Name = "nBayes";
            this.nBayes.Size = new System.Drawing.Size(85, 17);
            this.nBayes.TabIndex = 20;
            this.nBayes.TabStop = true;
            this.nBayes.Text = "Naive Bayes";
            this.nBayes.UseVisualStyleBackColor = true;
            this.nBayes.CheckedChanged += new System.EventHandler(this.Aglorithm_CheckedChanged);
            // 
            // nNet
            // 
            this.nNet.AutoSize = true;
            this.nNet.Location = new System.Drawing.Point(5, 57);
            this.nNet.Margin = new System.Windows.Forms.Padding(2);
            this.nNet.Name = "nNet";
            this.nNet.Size = new System.Drawing.Size(99, 17);
            this.nNet.TabIndex = 21;
            this.nNet.TabStop = true;
            this.nNet.Text = "Neural Network";
            this.nNet.UseVisualStyleBackColor = true;
            this.nNet.CheckedChanged += new System.EventHandler(this.Aglorithm_CheckedChanged);
            // 
            // compAll
            // 
            this.compAll.AutoSize = true;
            this.compAll.Location = new System.Drawing.Point(118, 57);
            this.compAll.Margin = new System.Windows.Forms.Padding(2);
            this.compAll.Name = "compAll";
            this.compAll.Size = new System.Drawing.Size(80, 17);
            this.compAll.TabIndex = 22;
            this.compAll.TabStop = true;
            this.compAll.Text = "Compare all";
            this.compAll.UseVisualStyleBackColor = true;
            this.compAll.CheckedChanged += new System.EventHandler(this.Aglorithm_CheckedChanged);
            // 
            // testPercentage
            // 
            this.testPercentage.AutoSize = true;
            this.testPercentage.Font = new System.Drawing.Font("Microsoft Sans Serif", 10.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.testPercentage.Location = new System.Drawing.Point(11, 349);
            this.testPercentage.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.testPercentage.Name = "testPercentage";
            this.testPercentage.Size = new System.Drawing.Size(223, 18);
            this.testPercentage.TabIndex = 25;
            this.testPercentage.Text = "Enter the size of the train dataset";
            // 
            // trainDatasetSize
            // 
            this.trainDatasetSize.Location = new System.Drawing.Point(9, 386);
            this.trainDatasetSize.Margin = new System.Windows.Forms.Padding(2);
            this.trainDatasetSize.Name = "trainDatasetSize";
            this.trainDatasetSize.Size = new System.Drawing.Size(55, 20);
            this.trainDatasetSize.TabIndex = 26;
            // 
            // minValue
            // 
            this.minValue.AutoSize = true;
            this.minValue.Location = new System.Drawing.Point(223, 386);
            this.minValue.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.minValue.Name = "minValue";
            this.minValue.Size = new System.Drawing.Size(44, 13);
            this.minValue.TabIndex = 27;
            this.minValue.Text = "Min: 5%";
            // 
            // maxValue
            // 
            this.maxValue.AutoSize = true;
            this.maxValue.Location = new System.Drawing.Point(317, 386);
            this.maxValue.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.maxValue.Name = "maxValue";
            this.maxValue.Size = new System.Drawing.Size(53, 13);
            this.maxValue.TabIndex = 28;
            this.maxValue.Text = "Max: 70%";
            // 
            // percentOftestSize
            // 
            this.percentOftestSize.AutoSize = true;
            this.percentOftestSize.Location = new System.Drawing.Point(91, 386);
            this.percentOftestSize.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.percentOftestSize.Name = "percentOftestSize";
            this.percentOftestSize.Size = new System.Drawing.Size(114, 13);
            this.percentOftestSize.TabIndex = 29;
            this.percentOftestSize.Text = "% of the whole dataset";
            // 
            // datasetWithPCA
            // 
            this.datasetWithPCA.AutoSize = true;
            this.datasetWithPCA.Location = new System.Drawing.Point(6, 19);
            this.datasetWithPCA.Name = "datasetWithPCA";
            this.datasetWithPCA.Size = new System.Drawing.Size(197, 17);
            this.datasetWithPCA.TabIndex = 30;
            this.datasetWithPCA.TabStop = true;
            this.datasetWithPCA.Text = "Legitimate transactions, applied PCA";
            this.datasetWithPCA.UseVisualStyleBackColor = true;
            this.datasetWithPCA.CheckedChanged += new System.EventHandler(this.DatasetType_CheckedChanged);
            // 
            // datasetGenerated
            // 
            this.datasetGenerated.AutoSize = true;
            this.datasetGenerated.Location = new System.Drawing.Point(6, 59);
            this.datasetGenerated.Name = "datasetGenerated";
            this.datasetGenerated.Size = new System.Drawing.Size(187, 17);
            this.datasetGenerated.TabIndex = 31;
            this.datasetGenerated.TabStop = true;
            this.datasetGenerated.Text = "Legitimate transactions, generated";
            this.datasetGenerated.UseVisualStyleBackColor = true;
            this.datasetGenerated.CheckedChanged += new System.EventHandler(this.DatasetType_CheckedChanged);
            // 
            // datasetType
            // 
            this.datasetType.AutoSize = true;
            this.datasetType.Font = new System.Drawing.Font("Microsoft Sans Serif", 11.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.datasetType.Location = new System.Drawing.Point(9, 443);
            this.datasetType.Name = "datasetType";
            this.datasetType.Size = new System.Drawing.Size(90, 18);
            this.datasetType.TabIndex = 32;
            this.datasetType.Text = "Dataset type";
            // 
            // groupBoxAlgorithms
            // 
            this.groupBoxAlgorithms.Controls.Add(this.selectedAlgBut);
            this.groupBoxAlgorithms.Controls.Add(this.decTree);
            this.groupBoxAlgorithms.Controls.Add(this.randForest);
            this.groupBoxAlgorithms.Controls.Add(this.nBayes);
            this.groupBoxAlgorithms.Controls.Add(this.nNet);
            this.groupBoxAlgorithms.Controls.Add(this.compAll);
            this.groupBoxAlgorithms.Location = new System.Drawing.Point(9, 223);
            this.groupBoxAlgorithms.Name = "groupBoxAlgorithms";
            this.groupBoxAlgorithms.Size = new System.Drawing.Size(364, 100);
            this.groupBoxAlgorithms.TabIndex = 33;
            this.groupBoxAlgorithms.TabStop = false;
            // 
            // selectedAlgBut
            // 
            this.selectedAlgBut.Location = new System.Drawing.Point(231, 45);
            this.selectedAlgBut.Name = "selectedAlgBut";
            this.selectedAlgBut.Size = new System.Drawing.Size(118, 40);
            this.selectedAlgBut.TabIndex = 23;
            this.selectedAlgBut.Text = "Select algorithm";
            this.selectedAlgBut.UseVisualStyleBackColor = true;
            this.selectedAlgBut.Click += new System.EventHandler(this.SelectedAlgBut_Click);
            // 
            // groupBoxDatasetType
            // 
            this.groupBoxDatasetType.Controls.Add(this.selectedDatasetType);
            this.groupBoxDatasetType.Controls.Add(this.datasetWithPCA);
            this.groupBoxDatasetType.Controls.Add(this.datasetGenerated);
            this.groupBoxDatasetType.Location = new System.Drawing.Point(12, 475);
            this.groupBoxDatasetType.Name = "groupBoxDatasetType";
            this.groupBoxDatasetType.Size = new System.Drawing.Size(358, 100);
            this.groupBoxDatasetType.TabIndex = 34;
            this.groupBoxDatasetType.TabStop = false;
            // 
            // selectedDatasetType
            // 
            this.selectedDatasetType.Location = new System.Drawing.Point(226, 30);
            this.selectedDatasetType.Name = "selectedDatasetType";
            this.selectedDatasetType.Size = new System.Drawing.Size(118, 35);
            this.selectedDatasetType.TabIndex = 32;
            this.selectedDatasetType.Text = "Select dataset type";
            this.selectedDatasetType.UseVisualStyleBackColor = true;
            this.selectedDatasetType.Click += new System.EventHandler(this.SelectedDatasetType_Click);
            // 
            // debug
            // 
            this.debug.Location = new System.Drawing.Point(394, 475);
            this.debug.Margin = new System.Windows.Forms.Padding(2);
            this.debug.Multiline = true;
            this.debug.Name = "debug";
            this.debug.Size = new System.Drawing.Size(669, 119);
            this.debug.TabIndex = 35;
            // 
            // Start
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ScrollBar;
            this.ClientSize = new System.Drawing.Size(1074, 670);
            this.Controls.Add(this.debug);
            this.Controls.Add(this.groupBoxDatasetType);
            this.Controls.Add(this.groupBoxAlgorithms);
            this.Controls.Add(this.datasetType);
            this.Controls.Add(this.percentOftestSize);
            this.Controls.Add(this.maxValue);
            this.Controls.Add(this.minValue);
            this.Controls.Add(this.trainDatasetSize);
            this.Controls.Add(this.testPercentage);
            this.Controls.Add(this.selAlg);
            this.Controls.Add(this.ClearResults);
            this.Controls.Add(this.datasetName);
            this.Controls.Add(this.testDataSetName);
            this.Controls.Add(this.algRes);
            this.Controls.Add(this.AlgResult);
            this.Controls.Add(this.CloseApp);
            this.Controls.Add(this.AddData);
            this.Controls.Add(this.ClearData);
            this.Controls.Add(this.FraudDet);
            this.Controls.Add(this.appName);
            this.Controls.Add(this.label1);
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "Start";
            this.Text = "CCFD";
            this.groupBoxAlgorithms.ResumeLayout(false);
            this.groupBoxAlgorithms.PerformLayout();
            this.groupBoxDatasetType.ResumeLayout(false);
            this.groupBoxDatasetType.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label appName;
        private System.Windows.Forms.Button FraudDet;
        private System.Windows.Forms.Button ClearData;
        private System.Windows.Forms.Button AddData;
        private System.Windows.Forms.Button CloseApp;
        private System.Windows.Forms.TextBox AlgResult;
        private System.Windows.Forms.Label algRes;
        private System.Windows.Forms.TextBox testDataSetName;
        private System.Windows.Forms.Label datasetName;
        private System.Windows.Forms.Button ClearResults;
        private System.Windows.Forms.Label selAlg;
        private System.Windows.Forms.RadioButton decTree;
        private System.Windows.Forms.RadioButton randForest;
        private System.Windows.Forms.RadioButton nBayes;
        private System.Windows.Forms.RadioButton nNet;
        private System.Windows.Forms.RadioButton compAll;
        private System.Windows.Forms.Label testPercentage;
        private System.Windows.Forms.TextBox trainDatasetSize;
        private System.Windows.Forms.Label minValue;
        private System.Windows.Forms.Label maxValue;
        private System.Windows.Forms.Label percentOftestSize;
        private System.Windows.Forms.RadioButton datasetWithPCA;
        private System.Windows.Forms.RadioButton datasetGenerated;
        private System.Windows.Forms.Label datasetType;
        private System.Windows.Forms.GroupBox groupBoxAlgorithms;
        private System.Windows.Forms.GroupBox groupBoxDatasetType;
        private System.Windows.Forms.Button selectedAlgBut;
        private System.Windows.Forms.Button selectedDatasetType;
        private System.Windows.Forms.TextBox debug;
    }
}

