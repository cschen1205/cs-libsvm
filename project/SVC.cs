using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibSvmSharp.libsvm;
using LibSvmSharp.metrics;

namespace LibSvmSharp
{
    public class SVC
    {
        private SVMParam _param;
        private bool _crossValidation;
        private SVMModel model;
        private bool quiet;

        public SVMParam Param
        {
            get { return _param; }
            set { _param = value; }
        }

        public bool CrossValidation
        {
            get { return _crossValidation; }
            set { _crossValidation = value; }
        }
        


        public void copy(SVC rhs)
        {
            _param = rhs._param == null ? null : (SVMParam)rhs._param.Clone();
            _crossValidation = rhs._crossValidation;
            model = rhs.model == null ? null : (SVMModel)rhs.model.Clone();
            quiet = rhs.quiet;
        }


        public Object Clone()
        {
            SVC clone = new SVC();
            clone.copy(this);

            return clone;
        }

        public SVC()
            : base()
        {
            init();
        }


        public SVMModel GetModel()
        {
            return model;
        }

        public SVMParam GetConfig()
        {
            return _param;
        }

        public void SetConfig(SVMParam parameters)
        {
            this._param = parameters;
        }

        public void SetModel(SVMModel model)
        {
            this.model = model;
        }

        public bool isQuiet()
        {
            return quiet;
        }

        public void setQuiet(bool quiet)
        {
            this.quiet = quiet;
        }

        public SVMType getSVMType()
        {
            if (_param.SVMType == SVMParam.SVM_TYPE_C_SVC)
            {
                return SVMType.C;
            }
            else {
                return SVMType.nu;
            }
        }

        public void setSVMType(SVMType type)
        {
            switch (type)
            {
                case SVMType.C:
                    _param.SVMType = SVMParam.SVM_TYPE_C_SVC;
                    break;
                case SVMType.nu:
                    _param.SVMType = SVMParam.SVM_TYPE_NU_SVC;
                    break;
            }
        }

        private void init()
        {
            _param = new SVMParam();
            // default values
            _param.SVMType = SVMParam.SVM_TYPE_C_SVC;
            _param.KernelType = SVMParam.KERNEL_TYPE_RBF;
            _param.Degree = 3;
            _param.Gamma = 0;   // 1/num_features
            _param.Coef0 = 0;
            _param.nu = 0.5;
            _param.CacheSizeInMB = 100;
            _param.C = 1;
            _param.Epsilon = 1e-3;
            _param.p = 0.1;
            _param.UseShrinkingHeuristic = true;
            _param.DoProbabilityEstimate = false;
            _param.NumberWeight = 0;
            _param.WeightLabel = new int[0];
            _param.Weight = new double[0];
            _crossValidation = false;

            libsvm.SVM.svm_set_print_string_function(null);
            quiet = false;
        }

        public SVMParam getParameters()
        {
            return _param;
        }

        public void setParameters(SVMParam parameters)
        {
            this._param = parameters;
        }

        private void info(String info)
        {

        }


        public double Predict(double[] x0)
        {
            int n = x0.Length;

            SVMNode[] x = new SVMNode[n];
            for (int j = 0; j < n; j++)
            {
                x[j] = new SVMNode();
                x[j].index = j + 1;
                x[j].value = x0[j];
            }

            double v = libsvm.SVM.svm_predict(model, x);


            return v;
        }


        public bool isInClass(double[] tuple)
        {
            double p = Predict(tuple);
            return p > 0;
        }

        // label: 1.0 : -1.0
        public AccuracyMetric Fit(List<KeyValuePair<double[], bool>> train_data, List<KeyValuePair<double[], bool>> test_data)
        {
            if(this.quiet)
            {
                libsvm.SVM.svm_set_print_string_function(new svm_print_null());
            }else{
                libsvm.SVM.svm_set_print_string_function(null);
            }

            List<double> vy = new List<double>();
            List<SVMNode[]> vx = new List<SVMNode[]>();
            int max_index = 0;

            int m = train_data.Count;

            for(int i = 0; i<m; ++i)
            {
                double[] x0 = train_data[i].Key;
                int n = x0.Length;

                vy.Add(train_data[i].Value ? 1 : -1);

                SVMNode[] x = new SVMNode[n];
                for(int j = 0; j<n; j++)
                {
                    x[j] = new SVMNode();
                    x[j].index = j+1;
                    x[j].value = x0[j];
                }

                if(n>0) max_index = Math.Max(max_index, x[n - 1].index);

                vx.Add(x);
            }

            SVMProblem prob = new SVMProblem();
            prob.ProblemSize = m;
            prob.x = new SVMNode[m][];
            for(int i = 0; i<m;i++)
                prob.x[i] = vx[i];
            prob.y = new double[m];
            for(int i = 0; i<m;i++)
                prob.y[i] = vy[i];

            if(_param.Gamma == 0 && max_index > 0)
                _param.Gamma = 1.0/max_index;


            model = libsvm.SVM.svm_train(prob, _param, (state) => { return false; });

            AccuracyMetric metric = new AccuracyMetric();
            metric.TrainAccuracy = GetAccuracy(train_data);
            metric.TestAccuracy = GetAccuracy(test_data);
            return metric;
        }

        private double GetAccuracy(List<KeyValuePair<double[], bool>> data)
        {
            double result = 0;
            foreach(KeyValuePair<double[], bool> entry in data)
            {
                double[] x = entry.Key;
                bool y = entry.Value;
                bool output = isInClass(x);
                result += y == output ? 1 : 0;
            }
            return result / data.Count;
        }

        public enum SVMType
        {
            C,
            nu
        }
    }
}
