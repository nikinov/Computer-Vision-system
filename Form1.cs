using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ONNXDetector;

namespace WindowsFormsTestApp
{
	public partial class Form1 : Form
	{
		private Detector _detector;
		public Form1()
		{
			InitializeComponent();
		}
		private void Form1_Load(object sender, EventArgs e)
		{
			_detector = new Detector();
			// prepare the prediction engine
			_detector.Init("../../model.onnx", "../../Labels.txt");
		}

		private void button1_Click(object sender, EventArgs e)
		{
			// initialise detector
			
			var bitmapImage = new Bitmap("../../BrownTest1.bmp");

			// predict on bitmap with a float array output
			var r = string.Join(@", ", _detector.GetPrediction(bitmapImage).Select(f => (f).ToString(@"0.000 %")));
			richTextBox1.AppendText(r + Environment.NewLine);

			// predict on bitmap with a string label output
			richTextBox1.AppendText(_detector.GetPredictionLabel(bitmapImage) + Environment.NewLine);
		}

		
	}
}
