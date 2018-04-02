using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PredictiveModels.Tools.Compute.SVM.libsvm
{
    public class SVR_Q : Kernel
    {
        private int l;
        private Cache cache;
	    private int[] sign;
        private int[] index;
        private int next_buffer;
        private float[][] buffer;
        private double[] QD;


        public SVR_Q(SVMProblem prob, SVMParam param)
            : base(prob.ProblemSize, prob.x, param)
        {
            l = prob.ProblemSize;
            cache = new Cache(l, (long)(param.CacheSizeInMB * (1 << 20)));
            QD = new double[2 * l];
            sign = new int[2 * l];
            index = new int[2 * l];
            for (int k = 0; k < l; k++)
            {
                sign[k] = 1;
                sign[k + l] = -1;
                index[k] = k;
                index[k + l] = k;
                QD[k] = kernel_function(k, k);
                QD[k + l] = QD[k];
            }
            buffer = new float[2][];
            for(int i=0; i < buffer.Length; ++i)
            {
                buffer[i] = new float[2 * l];
            }
            
            next_buffer = 0;
        }

        public override void swap_index(int i, int j)
        {
            do { int tmp = sign[i]; sign[i] = sign[j]; sign[j] = tmp; } while (false);
            do { int tmp = index[i]; index[i] = index[j]; index[j] = tmp; } while (false);
            do { double tmp = QD[i]; QD[i] = QD[j]; QD[j] = tmp; } while (false);
        }

        public override float[] get_Q(int i, int len)
        {
            float[][] data = new float[1][];
            int j, real_i = index[i];
            if (cache.get_data(real_i, data, l) < l)
            {
                for (j = 0; j < l; j++)
                    data[0][j] = (float)kernel_function(real_i, j);
            }

            // reorder and copy
            float[] buf = buffer[next_buffer];
            next_buffer = 1 - next_buffer;
            int si = sign[i];
            for (j = 0; j < len; j++)
                buf[j] = (float)si * sign[j] * data[0][index[j]];
            return buf;
        }

        public override double[] get_QD()
        {
            return QD;
        }
    }
}
