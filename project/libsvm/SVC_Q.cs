using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PredictiveModels.Tools.Compute.SVM.libsvm
{
    //
    // Q matrices for various formulations
    //
    public class SVC_Q : Kernel
    {

        private int[] y;
        private Cache cache;
	    private double[] QD;
    
        public SVC_Q(SVMProblem prob, SVMParam param, int[] y_)
            :base(prob.ProblemSize, prob.x, param)
        {
            y = (int[]) y_.Clone();
            cache = new Cache(prob.ProblemSize, (long)(param.CacheSizeInMB * (1 << 20)));
            QD = new double[prob.ProblemSize];
            for (int i = 0; i < prob.ProblemSize; i++)
                QD[i] = kernel_function(i, i);
        }

        public override float[] get_Q(int i, int len)
        {
            float[][] data = new float[1][];
            int start, j;
            if ((start = cache.get_data(i, data, len)) < len)
            {
                for (j = start; j < len; j++)
                    data[0][j] = (float)(y[i] * y[j] * kernel_function(i, j));
            }
            return data[0];
        }

        public override double[] get_QD()
        {
            return QD;
        }

        public override void swap_index(int i, int j)
        {
            cache.swap_index(i, j);
            base.swap_index(i, j);
            do { int tmp = y[i]; y[i] = y[j]; y[j] = tmp; } while (false);
            do { double tmp = QD[i]; QD[i] = QD[j]; QD[j] = tmp; } while (false);
        }
    }
}
