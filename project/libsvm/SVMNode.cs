using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibSvmSharp.libsvm
{
    public class SVMNode : ICloneable
    {
	    public int index;
        public double value;

        public void copy(SVMNode rhs)
        {
            index = rhs.index;
            value = rhs.value;
        }
    
        public object Clone()
        {
            SVMNode clone = new SVMNode();
            clone.copy(this);
            return clone;
        }
    }

}
