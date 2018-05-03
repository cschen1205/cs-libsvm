using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibSvmSharp.libsvm
{
    public abstract class Kernel : QMatrix
    {

        private SVMNode[][] x;
        private readonly double[] x_square;

        // svm_parameter
        private readonly int kernel_type;
        private readonly int degree;
        private readonly double gamma;
        private readonly double coef0;
        
        public override void swap_index(int i, int j)
        {
            do { SVMNode[] tmp = x[i]; x[i] = x[j]; x[j] = tmp; } while (false);
            if (x_square != null) do { double tmp = x_square[i]; x_square[i] = x_square[j]; x_square[j] = tmp; } while (false);
        }

        private static double powi(double basenumber, int times)
        {
            double tmp = basenumber, ret = 1.0;

            for (int t = times; t > 0; t /= 2)
            {
                if (t % 2 == 1) ret *= tmp;
                tmp = tmp * tmp;
            }
            return ret;
        }

        protected double kernel_function(int i, int j)
        {
            switch (kernel_type)
            {
                case SVMParam.KERNEL_TYPE_LINEAR:
                    return dot(x[i], x[j]);
                case SVMParam.KERNEL_TYPE_POLY:
                    return powi(gamma * dot(x[i], x[j]) + coef0, degree);
                case SVMParam.KERNEL_TYPE_RBF:
                    return Math.Exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
                case SVMParam.KERNEL_TYPE_SIGMOID:
                    return Math.Tanh(gamma * dot(x[i], x[j]) + coef0);
                case SVMParam.KERNEL_TYPE_PRECOMPUTED:
                    return x[i][(int)(x[j][0].value)].value;
                default:
                    return 0;   // java
            }
        }


        public Kernel(int l, SVMNode[][] x_, SVMParam param)
        {
            this.kernel_type = param.KernelType;
            this.degree = param.Degree;
            this.gamma = param.Gamma;
            this.coef0 = param.Coef0;

            x = (SVMNode[][]) x_.Clone();

            if (kernel_type == SVMParam.KERNEL_TYPE_RBF)
            {
                x_square = new double[l];
                for (int i = 0; i < l; i++)
                    x_square[i] = dot(x[i], x[i]);
            }
            else x_square = null;
        }

        static double dot(SVMNode[] x, SVMNode[] y)
        {
            double sum = 0;
            int xlen = x.Length;
            int ylen = y.Length;
            int i = 0;
            int j = 0;
            while (i < xlen && j < ylen)
            {
                if (x[i].index == y[j].index)
                    sum += x[i++].value * y[j++].value;
                else
                {
                    if (x[i].index > y[j].index)
                        ++j;
                    else
                        ++i;
                }
            }
            return sum;
        }

        public static double k_function(SVMNode[] x, SVMNode[] y,
                        SVMParam param)
        {
            switch (param.KernelType)
            {
                case SVMParam.KERNEL_TYPE_LINEAR:
                    return dot(x, y);
                case SVMParam.KERNEL_TYPE_POLY:
                    return powi(param.Gamma * dot(x, y) + param.Coef0, param.Degree);
                case SVMParam.KERNEL_TYPE_RBF:
                    {
                        double sum = 0;
                        int xlen = x.Length;
                        int ylen = y.Length;
                        int i = 0;
                        int j = 0;
                        while (i < xlen && j < ylen)
                        {
                            if (x[i].index == y[j].index)
                            {
                                double d = x[i++].value - y[j++].value;
                                sum += d * d;
                            }
                            else if (x[i].index > y[j].index)
                            {
                                sum += y[j].value * y[j].value;
                                ++j;
                            }
                            else
                            {
                                sum += x[i].value * x[i].value;
                                ++i;
                            }
                        }

                        while (i < xlen)
                        {
                            sum += x[i].value * x[i].value;
                            ++i;
                        }

                        while (j < ylen)
                        {
                            sum += y[j].value * y[j].value;
                            ++j;
                        }

                        return Math.Exp(-param.Gamma * sum);
                    }
                case SVMParam.KERNEL_TYPE_SIGMOID:
                    return Math.Tanh(param.Gamma * dot(x, y) + param.Coef0);
                case SVMParam.KERNEL_TYPE_PRECOMPUTED:
                    return x[(int)(y[0].value)].value;
                default:
                    return 0;   // java
            }
        }
    }
}
