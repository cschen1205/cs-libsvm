using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibSvmSharp.libsvm
{
    public class ONE_CLASS_Q : Kernel
    {
        private Cache cache;
	    private double[] QD;


        public ONE_CLASS_Q(SVMProblem prob, SVMParam param)
            : base(prob.ProblemSize, prob.x, param)
        {
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
                    data[0][j] = (float)kernel_function(i, j);
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
            do { double tmp = QD[i]; QD[i] = QD[j]; QD[j] = tmp; } while (false);
        }
    }

}
