using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CreditCardFraudDet
{
    public partial class DecisionTreePrunedAROC : Form
    {
        public DecisionTreePrunedAROC()
        {
            InitializeComponent();
        }

        public void ShowImage(Image image)
        {
            decisionTreeShow.SizeMode = PictureBoxSizeMode.Zoom;
            decisionTreeShow.Image = image;
        }
    }
}
