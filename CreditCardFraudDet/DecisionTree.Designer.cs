﻿
namespace CreditCardFraudDet
{
    partial class DecisionTreeBuilt
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
            this.decisionTreeShow = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.decisionTreeShow)).BeginInit();
            this.SuspendLayout();
            // 
            // decisionTreeShow
            // 
            this.decisionTreeShow.Location = new System.Drawing.Point(11, 12);
            this.decisionTreeShow.Name = "decisionTreeShow";
            this.decisionTreeShow.Size = new System.Drawing.Size(1051, 651);
            this.decisionTreeShow.TabIndex = 0;
            this.decisionTreeShow.TabStop = false;
            // 
            // DecisionTreeBuilt
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1074, 670);
            this.Controls.Add(this.decisionTreeShow);
            this.Name = "DecisionTreeBuilt";
            this.Text = "DecisionTreeBuilt";
            ((System.ComponentModel.ISupportInitialize)(this.decisionTreeShow)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox decisionTreeShow;
    }
}