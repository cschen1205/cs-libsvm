using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibSvmSharp.libsvm
{
    public class SVMParam : ICloneable
    {
	    /* svm_type */
	    public const int SVM_TYPE_C_SVC = 0;
        public const int SVM_TYPE_NU_SVC = 1;
        public const int SVM_TYPE_ONE_CLASS = 2;
        public const int SVM_TYPE_EPSILON_SVR = 3;
        public const int SVM_TYPE_NU_SVR = 4;

        /* kernel_type */
        public const int KERNEL_TYPE_LINEAR = 0;
        public const int KERNEL_TYPE_POLY = 1;
        public const int KERNEL_TYPE_RBF = 2;
        public const int KERNEL_TYPE_SIGMOID = 3;
        public const int KERNEL_TYPE_PRECOMPUTED = 4;

        public int SVMType { get; set; }
        public int KernelType { get; set; }
        public int Degree
        {
            get; set;
        }  // for poly
        public double Gamma { get; set; }    // for poly/rbf/sigmoid
        public double Coef0 { get; set; }    // for poly/sigmoid

        // these are for training only
        public double CacheSizeInMB { get; set; } // in MB
        public double Epsilon { get; set; }  // stopping actionselection
        public double C { get; set; }    // for C_SVC, EPSILON_SVR and NU_SVR
        public int NumberWeight { get; set; }       // for C_SVC
        public int[] WeightLabel { get; set; }  // for C_SVC
        public double[] Weight { get; set; }     // for C_SVC
        public double nu { get; set; }   // for NU_SVC, ONE_CLASS, and NU_SVR
        public double p { get; set; }    // for EPSILON_SVR
        public bool UseShrinkingHeuristic { get; set; }   // use the shrinking heuristics
        public bool DoProbabilityEstimate { get; set; } // do probability estimates

        public void copy(SVMParam rhs)
        {
            SVMType = rhs.SVMType;
            KernelType = rhs.KernelType;
            Degree = rhs.Degree;    // for poly
            Gamma = rhs.Gamma;  // for poly/rbf/sigmoid
            Coef0 = rhs.Coef0;  // for poly/sigmoid

            // these are for training only
            CacheSizeInMB = rhs.CacheSizeInMB; // in MB
            Epsilon = rhs.Epsilon;  // stopping actionselection
            C = rhs.C;  // for C_SVC, EPSILON_SVR and NU_SVR
            NumberWeight = rhs.NumberWeight;      // for C_SVC
            WeightLabel = (int[])rhs.WeightLabel.Clone();    // for C_SVC
            Weight = (double[]) rhs.Weight.Clone();        // for C_SVC
            nu = rhs.nu;    // for NU_SVC, ONE_CLASS, and NU_SVR
            p = rhs.p;  // for EPSILON_SVR
            UseShrinkingHeuristic = rhs.UseShrinkingHeuristic;  // use the shrinking heuristics
            DoProbabilityEstimate = rhs.DoProbabilityEstimate; // do probability estimates
        }

        public object Clone()
        {
            SVMParam clone = new SVMParam();
            clone.copy(this);
            return clone;
        }
    }

}
