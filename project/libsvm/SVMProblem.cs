﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibSvmSharp.libsvm
{
    public class SVMProblem 
    {
        public int ProblemSize { get; set; }
        public double[] y { get; set; }
        public SVMNode[][] x { get; set; }
    }
}
