using System;
using System.Collections.Generic;
using System.Linq;

namespace LibSvmSharp
{
    class Program
    {
        
        private static Random _random = new Random();
        public static double randn()
        {
            return _random.NextDouble();
        }

        public static double rand(double lower, double upper)
        {
            return randn() * (upper - lower) + lower;
        }


        static void Main(string[] args)
        {
            List<KeyValuePair<double[], double>> trainingBatch = new List<KeyValuePair<double[], double>>();
            // add some normal data
            for (int i = 0; i < 100; ++i)
            {
                trainingBatch.Add(new KeyValuePair<double[], double>(new double[] { randn() * 0.3 + 2, randn() * 0.3 + 2 }, -1.0));
                trainingBatch.Add(new KeyValuePair<double[], double>(new double[] { randn() * 0.3 - 2, randn() * 0.3 - 2 }, -1.0));
            }

            List<KeyValuePair<double[], double>> crossValidationBatch = new List<KeyValuePair<double[], double>>();
            // add some validation data
            for (int i = 0; i < 20; ++i)
            {
                crossValidationBatch.Add(new KeyValuePair<double[], double>(new double[] { randn() * 0.3 + 2, randn() * 0.3 + 2 }, -1.0));
                crossValidationBatch.Add(new KeyValuePair<double[], double>(new double[] { randn() * 0.3 - 2, randn() * 0.3 - 2 }, -1.0));
            }

            List<KeyValuePair<double[], double>> outliers = new List<KeyValuePair<double[], double>>();
            // add some outliers data
            for (int i = 0; i < 20; ++i)
            {
                outliers.Add(new KeyValuePair<double[], double>(new double[] { rand(-4, 4), rand(-4, 4) }, 1.0));
                outliers.Add(new KeyValuePair<double[], double>(new double[] { rand(-4, 4), rand(-4, 4) }, 1.0));
            }

            OneClassSVM algorithm = new OneClassSVM();
            algorithm.set_gamma(0.1);
            algorithm.set_nu(0.1);

            var batches = from p in trainingBatch
                          select p.Key;
            algorithm.Fit(batches.ToList());

            for (int i = 0; i < crossValidationBatch.Count; ++i)
            {
                double predicted = algorithm.IsOutlier(crossValidationBatch[i].Key) ? 1.0 : -1.0;
                Console.WriteLine("predicted: " + predicted + "\texpected: " + crossValidationBatch[i].Value);
            }

            for (int i = 0; i < outliers.Count; ++i)
            {
                double predicted = algorithm.IsOutlier(crossValidationBatch[i].Key) ? 1.0 : -1.0;
                Console.WriteLine("predicted: " + predicted + "\texpected: " + outliers[i].Value);
            }

        }

        private List<double[]> GetInputVectors(List<KeyValuePair<double[], double>> samples)
        {
            List<double[]> inputVectors = new List<double[]>();
            foreach (KeyValuePair<double[], double> point in samples)
            {
                inputVectors.Add(point.Key);
            }
            return inputVectors;
        }
    }
}
