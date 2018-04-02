using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PredictiveModels.Tools.Compute.SVM.libsvm
{
    public class SVMModel : ICloneable
    {
	    private SVMParam _param;    // parameter
        private int _numberClass;        // number of classes, = 2 in regression/one class svm
        private int _totalNumberSV;            // total #SV
        private SVMNode[][] _SV;    // SVs (SV[l])
        private double[][] _coefSV;    // coefficients for SVs in decision functions (sv_coef[k-1][l])
        private double[] _rho;        // constants in decision functions (rho[k*(k-1)/2])
        private double[] _probA;         // pariwise probability information
        private double[] _probB;
        private int[] _indicesSV;       // sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set

        // for classification only

        private int[] _label;        // label of each class (label[k])
        private int[] _numberSV4EachClass;        // number of SVs for each class (nSV[k])
                                 // nSV[0] + nSV[1] + ... + nSV[k-1] = l

        public int[] NumberSV4EachClass
        {
            get { return _numberSV4EachClass; }
            set { _numberSV4EachClass = value; }
        }

        public int[] Label
        {
            get { return _label; }
            set { _label = value; }
        }

        public double[] ProbA
        {
            get { return _probA; }
            set { _probA = value; }
        }

        public double[] ProbB
        {
            get { return _probB; }
            set { _probB = value; }
        }

        public int[] IndicesSV
        {
            get { return _indicesSV; }
            set { _indicesSV = value; }
        }

        public double[] Rho
        {
            get { return _rho; }
            set { _rho = value; }
        }

        public double[][] CoefSV
        {
            get { return _coefSV; }
            set { _coefSV = value; }
        }

        public SVMNode[][] SV
        {
            get { return _SV; }
            set { _SV = value; }
        }

        public int TotalNumberSV
        {
            get { return _totalNumberSV; }
            set { _totalNumberSV = value; }
        }

        public SVMParam Param
        {
            get { return _param; }
            set { _param = value; }
        }

        public int NumberClass
        {
            get { return _numberClass; }
            set { _numberClass = value; }
        }
                                 
        public object Clone()
        {
            SVMModel clone = new SVMModel();
            clone.Copy(this);

            return clone;
        }

        public void Copy(SVMModel rhs)
        {
            _param = rhs._param;
            _numberClass = rhs._numberClass;
            _totalNumberSV = rhs._totalNumberSV;            // total #SV
            _SV = new SVMNode[rhs._SV.Length][];    // SVs (SV[l])

            for (int i = 0; i < rhs._SV.Length; ++i)
            {
                _SV[i] = new SVMNode[rhs._SV[i].Length];
                for (int j = 0; j < rhs._SV[i].Length; ++j)
                {
                    _SV[i][j] = (SVMNode)rhs._SV[i][j].Clone();
                }
            }

            _coefSV = new double[rhs._coefSV.Length][];    // coefficients for SVs in decision functions (sv_coef[k-1][l])
            for (int i = 0; i < rhs._coefSV.Length; ++i)
            {
                _coefSV[i] = (double[])rhs._coefSV[i].Clone();
            }

            _rho = rhs._rho == null ? null : (double[]) rhs._rho.Clone();        // constants in decision functions (rho[k*(k-1)/2])
            ProbA = rhs.ProbA == null ? null : (double[]) rhs.ProbA.Clone();         // pariwise probability information
            ProbB = rhs.ProbB == null ? null : (double[]) rhs.ProbB.Clone();
            _indicesSV = rhs._indicesSV == null ? null : (int[]) rhs._indicesSV.Clone();       // sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set

            // for classification only

            Label = rhs.Label == null ? null : (int[]) rhs.Label.Clone();        // label of each class (label[k])
            _numberSV4EachClass = rhs._numberSV4EachClass == null ? null : (int[]) rhs._numberSV4EachClass.Clone();        // number of SVs for each class (nSV[k])
        }
    }
}
