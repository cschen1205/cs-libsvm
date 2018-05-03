using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibSvmSharp.libsvm
{
    public interface svm_print_interface
    {
        void print(String s);
    }

    public class svm_print_stdout : svm_print_interface
    {
        public void print(String s)
        {
            Console.WriteLine(s);
        }
    }

    public class svm_print_null : svm_print_interface
    {
        public void print(string s)
        {

        }
    }
}
