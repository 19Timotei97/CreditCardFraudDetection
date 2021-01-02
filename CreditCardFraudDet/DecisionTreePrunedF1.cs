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
    public partial class DecisionTreePrunedF1 : Form
    {
        public DecisionTreePrunedF1()
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
