using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PredictiveModels.Tools.Commons.Compute;
using PredictiveModels.Tools.Commons.Compute.Models;
using PredictiveModels.Tools.Commons.Data;
using PredictiveModels.Tools.Compute.SVM.libsvm;

namespace PredictiveModels.Tools.Compute.SVM
{
    public class SVRMemento : INotifyPropertyChanged
    {
        #region [member variables]
        private bool _isValid = false;
        private SVR[] _model = new SVR[0];
        private Levels _dataLevelTable = new Levels();
        private PredictionConfig _predictionConfig = new PredictionConfig();
        private double _trainingError = 0.0;
        private List<DataColumnModel> _dataSchema = new List<DataColumnModel>();
        private SVRTrainParam _trainingConfig = new SVRTrainParam();
        private string _rBFModel = "";
        private int _inputDimension = 0;
        private int _outputDimension = 0;
        private string _title = "";
        private Dictionary<int, DataColumnStatistics> _columnScalings = new Dictionary<int, DataColumnStatistics>();
        #endregion

        #region [events]
        public event PropertyChangedEventHandler PropertyChanged;
        #endregion



        #region [constructors]
        public SVRMemento(SVRPredictionModel model)
        {
            if (model != null)
            {
                _dataLevelTable = model.DataLevelTable;
                _predictionConfig = model.PredictionConfig;
                _trainingError = model.TrainingError;
                _dataSchema = model.DataSchema;
                _trainingConfig = model.TrainingConfig;
                _inputDimension = model.InputDimension;
                _outputDimension = model.OutputDimension;
                _columnScalings = model.ColumnScalings;
                _title = model.Title;

                _model = model.SVR;
                _isValid = true;
            } else
            {
                _isValid = false;
            }
            
        }

        public SVRMemento()
        {

        }
        #endregion

        #region [properties]
        public Levels DataLevelTable
        {
            get
            {
                return _dataLevelTable;
            }
            set
            {
                if (_dataLevelTable != value)
                {
                    _dataLevelTable = value;
                    NotifyPropertyChanged("DataLevelTable");
                }
            }
        }

        public string Title
        {
            get { return _title; }
            set
            {
                if(_title != value)
                {
                    _title = value;
                    NotifyPropertyChanged("Title");
                }
            }
        }

        public PredictionConfig PredictionConfig
        {
            get
            {
                return _predictionConfig;
            }
            set
            {
                if(_predictionConfig != value)
                {
                    _predictionConfig = value;
                    NotifyPropertyChanged("PredictionConfig");
                }
            }
        }

        public double TrainingError
        {
            get
            {
                return _trainingError;
            }
            set
            {
                if(_trainingError != value)
                {
                    _trainingError = value;
                    NotifyPropertyChanged("TrainingError");
                }
            }
        }

        public List<DataColumnModel> DataSchema
        {
            get
            {
                return _dataSchema;
            }
            set
            {
                if(_dataSchema != value)
                {
                    _dataSchema = value;
                    NotifyPropertyChanged("DataSchema");
                }
            }
        }

        public SVRTrainParam TrainingConfig
        {
            get
            {
                return _trainingConfig;
            }
            set
            {
                if(_trainingConfig != value)
                {
                    _trainingConfig = value;
                    NotifyPropertyChanged("TrainingConfig");
                }
            }
        }

        public string RBFModel
        {
            get
            {
                return _rBFModel;
            }
            set
            {
                if(_rBFModel != value)
                {
                    _rBFModel = value;
                    NotifyPropertyChanged("RBFModel");
                }
            }
        }

        public int InputDimension
        {
            get
            {
                return _inputDimension;
            }
            set
            {
                if(_inputDimension != value)
                {
                    _inputDimension = value;
                    NotifyPropertyChanged("InputDimension");
                }
            }
        }

        public int OutputDimension
        {
            get
            {
                return _outputDimension;
            }
            set
            {
                if(_outputDimension != value)
                {
                    _outputDimension = value;
                    NotifyPropertyChanged("OutputDimension");
                }
            }
        }

        public Dictionary<int, DataColumnStatistics> ColumnScalings
        {
            get
            {
                return _columnScalings;
            }
            set
            {
                if(_columnScalings != value)
                {
                    _columnScalings = value;
                    NotifyPropertyChanged("ColumnScalings");
                }
            }
        }

        public SVR[] Model
        {
            get { return _model; }
            set {
                if(_model != value)
                {
                    _model = value;
                    NotifyPropertyChanged("Model");
                }
            }
        }
        public bool IsValid
        {
            get
            {
                return _isValid;
            }
            set
            {
                if(_isValid != value)
                {
                    _isValid = value;
                    NotifyPropertyChanged("IsValid");
                }
            }
        }
        #endregion

        #region [internal methods]
        protected void NotifyPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        #endregion

        #region [exposed methods]
        public SVRPredictionModel Convert2Model()
        {
            if (_isValid)
            {
                SVRPredictionModel model = new SVRPredictionModel();

                model.PredictionConfig = _predictionConfig;
                model.DataLevelTable = _dataLevelTable;
                model.TrainingError = _trainingError;
                model.DataSchema = _dataSchema;
                model.TrainingConfig = _trainingConfig;
                model.InputDimension = _inputDimension;
                model.OutputDimension = _outputDimension;
                model.ColumnScalings = _columnScalings;
                model.Title = _title;

                model.SVR = _model;

                model.UpdateStates();

                return model;
            } else
            {
                return null;
            }
        }
        #endregion
    }
}
