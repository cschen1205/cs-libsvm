using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using PredictiveModels.Tools.Commons.Compute;
using PredictiveModels.Tools.Commons.Compute.Models;

namespace PredictiveModels.Tools.Compute.SVM
{
    public class SVRPredictionModel : SimplePredictionModel
    {
        private SVR[] _svr;
        private int _outputDimension;
        private int _inputDimension;

        public int OutputDimension {  get { return _outputDimension; } set { _outputDimension = value; } }
        public int InputDimension { get { return _inputDimension; } set { _inputDimension = value; } }

        private SVRTrainParam _trainingConfig = new SVRTrainParam();

        public SVRTrainParam TrainingConfig
        {
            get { return _trainingConfig; }
            set { _trainingConfig = value; }
        }

        public SVR[] SVR
        {
            get { return _svr; }
            set { _svr = value; }
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
        

        public SVRPredictionModel()
        {
            Title = "SVM (Support Vector Machine)";
        }

        public override void Build(PredictionDataSet ds)
        {
            _DebugFormat("Start Build");
            base.Build(ds);
            
            int rowCount = ds.RowCount;

            bool flattened = _trainingConfig.FlattenData;

            _inputDimension = ds.GetInputDataPointDimension(flattened);
            _outputDimension = ds.GetOutputDataPointDimension(flattened);

            _svr = new SVR[_outputDimension];

            for(int k=0; k < _outputDimension; ++k)
            {
                _svr[k] = new SVR();

                List<KeyValuePair<double[], double>> lds = new List<KeyValuePair<double[], double>>();

                for (int rowIndex = 0; rowIndex < rowCount; ++rowIndex)
                {
                    if (ds.GetOutput(rowIndex, flattened).ContainsMissingValues) continue;

                    IDataPoint input1 = ds.GetInput(rowIndex, flattened);
                    var input2 = new double[_inputDimension];

                    for (var i = 0; i < _inputDimension; i++)
                    {
                        input2[i] = input1[i];
                    }

                    IDataPoint output1 = ds.GetOutput(rowIndex, flattened);

                    lds.Add(new KeyValuePair<double[], double>(input2, output1[k]));
                }

                _svr[k].Learn(lds, (state) => {
                    return false;
                });
            }

            _trainingError = GetMSE(ds, TrainingConfig.FlattenData);
            
        }

        public void UpdateStates()
        {
            

        }

        public override PredictionDataSet Predict(DataTable table, string missingValueSymbol)
        {
            PredictionDataSet result = base.Predict(table, missingValueSymbol);

            try {
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

                    double[] outputValues = new double[_outputDimension];

                    for (var i = 0; i < _outputDimension; ++i)
                    {
                        outputValues[i] = _svr[i].evaluate(input2);
                    }


                    result.OverwriteOutput(result.ConvertToOutputDataPoint(outputValues, flattened), rowIndex);
                }
            }catch(Exception exception)
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
            

            double[] predictValues = new double[_outputDimension];

            for (int i = 0; i < _outputDimension; ++i)
            {
                predictValues[i] = _svr[i].evaluate(input2);
            }

            IDataPoint predict1 = ConvertToOutputDataPoint(predictValues, TrainingConfig.FlattenData);

            return predict1;
        }

        public override Dictionary<string, object> Properties()
        {
            return _trainingConfig.Properties();
        }
    }
}
