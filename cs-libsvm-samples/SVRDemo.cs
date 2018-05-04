using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibSvmSharp
{
    class SVRDemo
    {
        private static Random random = new Random();

        public static void SolveRegressionProblem()
        {
            SVR svr = new SVR();


            List<KeyValuePair<double[], double>> data = new List<KeyValuePair<double[], double>>();
            for(int i=0; i < 100; ++i)
            {
                double[] x = new double[2];
                x[0] = i / 50.0;
                x[1] = (i+1) / 100.0;
                double y = Math.Sin((x[0] + x[1]) * Math.PI);
                data.Add(new KeyValuePair<double[], double>(x, y));
            }

            var tuple = TrainTestSplit(data);
            var train_data = tuple.Item1;
            var test_data = tuple.Item2;

            var metric = svr.Fit(train_data, test_data);

            Console.WriteLine("Train MSE: {0}, Test MSE: {1}", metric.TrainMSE, metric.TestMSE);
            
            foreach(KeyValuePair<double[], double> entry in test_data.GetRange(0, 10)) {
                double[] x = entry.Key;
                double y = entry.Value;
                double output = svr.Predict(x);
                Console.WriteLine("Predicted: {0} Actual: {1}", output, y);
            }
        }

        private static Tuple<List<T>, List<T>> TrainTestSplit<T>(List<T> a, double test_size=0.2)
        {
            Shuffle(a);
            int splitIndex = (int)(a.Count * test_size);
            List<T> test_data = new List<T>();
            List<T> train_data = new List<T>();
            for(int i=0; i < a.Count; ++i)
            {
                if(i < splitIndex)
                {
                    test_data.Add(a[i]);
                } else
                {
                    train_data.Add(a[i]);
                }
            }
            return new Tuple<List<T>, List<T>>(train_data, test_data);
        }

        private static void Shuffle<T>(List<T> a)
        {
            int i = 0; 
            while(i < a.Count)
            {
                int j = random.Next(i + 1);
                Swap(a, i++, j);
            }
        }

        private static void Swap<T>(List<T> a, int i, int j)
        {
            T temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
}
