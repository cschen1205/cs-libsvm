using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibSvmSharp
{
    class SVCDemo
    {
        private static Random random = new Random();

        public static void SolveBinaryClassificationProblem()
        {
            SVC svc = new SVC();


            List<KeyValuePair<double[], bool>> data = new List<KeyValuePair<double[], bool>>();
            for(int i=0; i < 100; ++i)
            {
                double[] x = new double[2];
                x[0] = i / 50.0;
                x[1] = (i+1) / 100.0;
                bool y = Math.Sin((x[0] + x[1]) * Math.PI) > 0;
                data.Add(new KeyValuePair<double[], bool>(x, y));
            }

            var tuple = TrainTestSplit(data);
            var train_data = tuple.Item1;
            var test_data = tuple.Item2;

            var metric = svc.Fit(train_data, test_data);

            Console.WriteLine("Train Acc: {0}, Test Acc: {1}", metric.TrainAccuracy, metric.TestAccuracy);
            
            foreach(KeyValuePair<double[], bool> entry in test_data.GetRange(0, 10)) {
                double[] x = entry.Key;
                bool y = entry.Value;
                double output = svc.Predict(x);
                bool prediction = output > 0;
                Console.WriteLine("Predicted: {0} Actual: {1}", prediction, y);
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
