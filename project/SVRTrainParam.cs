using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PredictiveModels.Tools.Compute.SVM
{
    public class SVRTrainParam : INotifyPropertyChanged
    {
        private bool _flattened = true;
        public bool FlattenData
        {
            get { return _flattened; }
            set
            {
                if (_flattened != value)
                {
                    _flattened = value;
                    NotifyPropertyChanged("FlattenData");
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void NotifyPropertyChanged(string name)
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(name));
            }
        }



        public void Copy(SVRTrainParam rhs)
        {
            _flattened = rhs._flattened;
        }

        public SVRTrainParam Clone()
        {
            SVRTrainParam clone = new SVRTrainParam();
            clone.Copy(this);
            return clone;
        }

        public Dictionary<string, object> Properties()
        {
            return new Dictionary<string, object>();
        }
    }
}
