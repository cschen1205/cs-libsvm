using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PredictiveModels.Tools.Commons.Compute;
using PredictiveModels.Tools.Commons.Compute.Models;

namespace PredictiveModels.Tools.Compute.SVM
{
    public class OneClassSVMPredictionModel : SimplePredictionModel
    {
        private OneClassSVM _oneClass;
        private int _inputDimension;
        
        public int InputDimension { get { return _inputDimension; } set { _inputDimension = value; } }

        private SVCTrainParam _trainingConfig = new SVCTrainParam();

        public SVCTrainParam TrainingConfig
        {
            get { return _trainingConfig; }
            set { _trainingConfig = value; }
        }



        public Stream GenerateStreamFromString(string s)
        {
            MemoryStream stream = new MemoryStream();
            StreamWriter writer = new StreamWriter(stream);
            writer.Write(s);
            writer.Flush();
            stream.Position = 0;
            return stream;
        }


        public OneClassSVMPredictionModel()
        {

        }

        public override void Build(PredictionDataSet ds)
        {
            _DebugFormat("Start Build");
            base.Build(ds);

            int rowCount = ds.RowCount;

            bool flattened = _trainingConfig.FlattenData;

            _inputDimension = ds.GetInputDataPointDimension(flattened);

            _oneClass = new OneClassSVM();

            List<double[]> lds = new List<double[]>();

            for (int rowIndex = 0; rowIndex < rowCount; ++rowIndex)
            {
                IDataPoint input1 = ds.GetInput(rowIndex, flattened);
                var input2 = new double[_inputDimension];

                for (var i = 0; i < _inputDimension; i++)
                {
                    input2[i] = input1[i];
                }

                lds.Add(input2);
            }

            _oneClass.Learn(lds, (state) =>
            {
                if (ShouldTerminateHandler != null && ShouldTerminateHandler(state))
                {
                    return true;
                }
                return false;
            });

            _trainingError = GetMSE(ds, TrainingConfig.FlattenData);

        }

        public void UpdateStates()
        {


        }

        public override PredictionDataSet Predict(DataTable table, string missingValueSymbol)
        {
            PredictionDataSet result = base.Predict(table, missingValueSymbol);

            try
            {
                bool flattened = _trainingConfig.FlattenData;

                int rowCount = result.RowCount;
                for (int rowIndex = 0; rowIndex < rowCount; ++rowIndex)
                {
                    var input2 = new double[_inputDimension];
                    IDataPoint input1 = result.GetInput(rowIndex, flattened);
                    for (var i = 0; i < _inputDimension; i++)
                    {
                        input2[i] = input1[i];
                    }

                    double[] outputValues = new double[1];

                    outputValues[0] = _oneClass.IsOutlier(input2) ? 1 : 0;
                    
                    result.OverwriteOutput(result.ConvertToOutputDataPoint(outputValues, flattened), rowIndex);
                }
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception.Message);
            }

            return result;
        }


        public override double GetMSE(IDataPoint input1, IDataPoint output1)
        {
            IDataPoint predict1 = Predict(input1);
            return GetMSE4Prediction(predict1, output1);
        }

        public IDataPoint Predict(IDataPoint input1)
        {
            var input2 = new double[_inputDimension];

            for (var i = 0; i < _inputDimension; i++)
            {
                input2[i] = input1[i];
            }


            double[] predictValues = new double[1];

            predictValues[0] = _oneClass.IsOutlier(input2) ? 1 : 0;

            IDataPoint predict1 = ConvertToOutputDataPoint(predictValues, TrainingConfig.FlattenData);

            return predict1;
        }

        public override Dictionary<string, object> Properties()
        {
            return _trainingConfig.Properties();
        }
    }
}
